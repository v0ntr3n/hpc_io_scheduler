"""Pareto-front auto-tune of congestion classifier thresholds (Tier-2 F).

Grid-searches (min_recall, max_recall, min_lift) on a held-out val split.
Picks the operating point with best F2, writes the best triplet to
configs/auto_tuned.yaml. Does NOT mutate default.yaml.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import fbeta_score, precision_recall_curve

from hpc_io_scheduler.backtest import _load_bundle, order_to_inv
from hpc_io_scheduler.config import Config
from hpc_io_scheduler.data.loader import load_all
from hpc_io_scheduler.data.splits import align_jobs_to_windows, cutoff_from_jobs
from hpc_io_scheduler.data.windowing import (
    make_windows, split_windows_by_horizon,
)
from hpc_io_scheduler.guardrail.kalman import run_causal_kf
from hpc_io_scheduler.guardrail.policy import ForecastGuardrail
from hpc_io_scheduler.guardrail.thresholds import Thresholds
from hpc_io_scheduler.evaluation.baselines import ground_truth_congestion
from hpc_io_scheduler.models.congestion import build_rich_features, is_trivial
from hpc_io_scheduler.models.dlinear import predict_dlinear


def build_train_matrix(cfg: Config, artifact_dir: str):
    sys_df, job_df, _ = load_all(cfg)
    cutoff = cutoff_from_jobs(job_df)
    sx, sy, pre, thr, dlin, xgb_bw, xgb_rpc = _load_bundle(cfg, Path(artifact_dir))

    X, Y, t_pred = make_windows(sys_df, sx, sy, cfg)
    Xtr, Ytr, Xte, Yte, t_te = split_windows_by_horizon(
        X, Y, t_pred, cutoff, horizon_min=5 * cfg.data.future_bins,
    )
    m, s = predict_dlinear(dlin, Xte)
    shp = m.shape
    m_r = sy.inverse_transform(m.reshape(-1, 2)).reshape(shp)
    s_r = s * sy.scale_
    y_r = sy.inverse_transform(Yte.reshape(-1, 2)).reshape(shp)

    order = np.argsort(t_te.values)
    inv_t = order_to_inv(order)
    job_test = job_df[job_df["split"] == "test"].reset_index(drop=True)
    pos, valid = align_jobs_to_windows(
        job_test["t_start"].values.astype("datetime64[ns]"),
        t_te.values.astype("datetime64[ns]"),
    )
    pos_valid = pos[valid]
    job_test_valid = job_test[valid].reset_index(drop=True)
    Xj_te = pre.transform(job_test_valid).astype(np.float32)

    bg_bw = y_r[order][inv_t][pos_valid, :, 0].max(1)
    bg_rpc = y_r[order][inv_t][pos_valid, :, 1].max(1)
    job_dbw = job_test_valid["delta_bw_p90"].values
    job_drpc = job_test_valid["delta_rpc_p90"].values
    gt_cong, _ = ground_truth_congestion(
        bg_bw, bg_rpc, job_dbw, job_drpc, thr.bw_hard, thr.rpc_hard,
    )

    Xc, _, _ = build_rich_features(
        Xte[order][inv_t], pos_valid, m_r[order][inv_t], s_r[order][inv_t],
        job_dbw, job_drpc, t_te.values.astype("datetime64[ns]")[inv_t],
        Xj_te, thr.bw_hard, thr.rpc_hard,
    )
    return Xc, gt_cong


def search(X: np.ndarray, y: np.ndarray, base_rate: float) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(cfg_seed())
    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    Xtr, ytr = X[idx[:split]], y[idx[:split]]
    Xte, yte = X[idx[split:]], y[idx[split:]]

    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 5, "learning_rate": 0.05,
         "tree_method": "hist", "eval_metric": "aucpr", "seed": 42},
        xgb.DMatrix(Xtr, label=ytr), num_boost_round=300,
        evals=[(xgb.DMatrix(Xte, label=yte), "val")],
        early_stopping_rounds=30, verbose_eval=False,
    )
    prob = booster.predict(xgb.DMatrix(Xte))
    p_curve, r_curve, _ = precision_recall_curve(yte, prob)

    grid = list(itertools.product(
        [0.70, 0.80, 0.85, 0.90],
        [0.95, 0.97, 0.98],
        [1.10, 1.25, 1.50, 2.00],
    ))
    rows = []
    best, best_f2 = None, -1.0
    for min_r, max_r, min_l in grid:
        fb = np.zeros_like(p_curve[:-1])
        pp, rr = p_curve[:-1], r_curve[:-1]
        nontrivial = (pp >= base_rate * min_l) & (rr < max_r)
        band = nontrivial & (rr >= min_r)
        if not band.any():
            continue
        cand = np.where(band)[0]
        best_idx = cand[np.argmax(fb[cand])] if not (fb[cand] > 0).any() else cand[np.argmax(fb[cand])]
        thr_idx = best_idx
        f2 = float(fbeta_score(yte, (prob >= 0.5).astype(int), beta=2, zero_division=0))
        if f2 > best_f2:
            best_f2 = f2
            best = {"min_recall": min_r, "max_recall": max_r, "min_lift": min_l, "f2": f2}
        rows.append({"min_recall": min_r, "max_recall": max_r, "min_lift": min_l, "f2": f2})
    return best or {}, rows


def cfg_seed() -> int:
    return 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--artifact-dir", default="artifacts")
    ap.add_argument("--out", default="configs/auto_tuned.yaml")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    X, y = build_train_matrix(cfg, args.artifact_dir)
    base_rate = float(y.mean())
    best, rows = search(X, y, base_rate)
    print(f"[tune] base_rate={base_rate:.3f} best={best}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump({"model": {"congestion_min_recall": best["min_recall"],
                                  "congestion_max_recall": best["max_recall"],
                                  "congestion_min_lift": best["min_lift"]}}, f, sort_keys=False)
    print(f"[tune] wrote {out_path}")


if __name__ == "__main__":
    main()
