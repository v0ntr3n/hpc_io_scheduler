"""Rolling-window backtest (Tier-2 K).

For each (train_window, test_window) pair on the system timeseries, refits
DLinear + XGB on the train window and evaluates on the test window. Logs
the degradation curve to reports/rolling_backtest.csv.

Window stride defaults to 1 day; test window = 6 hours. Configurable.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from hpc_io_scheduler.config import Config
from hpc_io_scheduler.data.loader import load_all
from hpc_io_scheduler.data.splits import cutoff_from_jobs
from hpc_io_scheduler.data.windowing import (
    fit_system_scalers, make_windows, split_windows_by_horizon,
)
from hpc_io_scheduler.evaluation.baselines import ground_truth_congestion
from hpc_io_scheduler.evaluation.metrics import decision_metrics
from hpc_io_scheduler.guardrail.kalman import run_causal_kf
from hpc_io_scheduler.guardrail.policy import ForecastGuardrail
from hpc_io_scheduler.guardrail.thresholds import Thresholds, compute_thresholds
from hpc_io_scheduler.models.dlinear import train_dlinear, predict_dlinear
from hpc_io_scheduler.models.xgb_delta import train_xgb, predict_xgb


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    rmse_bw: float
    rmse_rpc: float
    decision_acc: float
    false_submit_rate: float
    n_test: int
    train_sec: float


def fold_iter(sys_df: pd.DataFrame, cfg: Config, train_days: int, test_hours: int):
    train = pd.Timedelta(days=train_days)
    test = pd.Timedelta(hours=test_hours)
    t0 = sys_df["bin"].min()
    t_end = sys_df["bin"].max()
    i = 0
    while t0 + train + test <= t_end:
        tr_end = t0 + train
        te_end = tr_end + test
        yield i, t0, tr_end, te_end
        t0 = t0 + test
        i += 1


def run_fold(
    fold: int, t0: pd.Timestamp, tr_end: pd.Timestamp, te_end: pd.Timestamp,
    sys_df: pd.DataFrame, job_df: pd.DataFrame, cfg: Config,
) -> FoldResult:
    sub_sys = sys_df[(sys_df["bin"] >= t0) & (sys_df["bin"] < te_end)].copy()
    sub_job = job_df[
        (job_df["t_start"] >= t0) & (job_df["t_start"] < te_end)
    ].copy()
    if len(sub_job) < 100 or len(sub_sys) < cfg.data.past_bins + cfg.data.future_bins:
        return None

    sub_job["split"] = np.where(sub_job["t_start"] < tr_end, "train", "test")

    t0_t = time.time()
    sx, sy = fit_system_scalers(sub_sys, tr_end)
    X, Y, t_pred = make_windows(sub_sys, sx, sy, cfg)
    Xtr, Ytr, Xte, Yte, t_te = split_windows_by_horizon(
        X, Y, t_pred, tr_end, horizon_min=5 * cfg.data.future_bins,
    )
    if len(Xtr) == 0 or len(Xte) == 0:
        return None

    dlin, _ = train_dlinear(Xtr, Ytr, cfg, device="cpu")
    m, _ = predict_dlinear(dlin, Xte, device="cpu")
    shp = m.shape
    m_r = sy.inverse_transform(m.reshape(-1, 2)).reshape(shp)
    y_r = sy.inverse_transform(Yte.reshape(-1, 2)).reshape(shp)
    rmse_bw = float(np.sqrt(np.mean((m_r[..., 0] - y_r[..., 0]) ** 2)))
    rmse_rpc = float(np.sqrt(np.mean((m_r[..., 1] - y_r[..., 1]) ** 2)))

    job_train = sub_job[sub_job["split"] == "train"]
    job_test = sub_job[sub_job["split"] == "test"]
    from hpc_io_scheduler.data.splits import fit_transform_jobs
    Xj_tr, Xj_te, pre = fit_transform_jobs(
        job_train.reset_index(drop=True), job_test.reset_index(drop=True),
    )
    groups = job_train["id_job_norm"].astype(str).str.split("_").str[0].values
    w = np.where(job_train["dnn_label"] != "unlabeled", 5.0, 1.0)
    if len(np.unique(groups)) < cfg.model.xgb_n_splits:
        return None
    xgb_bw, _ = train_xgb(Xj_tr, job_train["delta_bw_p90"].values, w, groups, cfg)
    xgb_rpc, _ = train_xgb(Xj_tr, job_train["delta_rpc_p90"].values, w, groups, cfg)
    thr = compute_thresholds(sub_sys, tr_end, cfg.guardrail)
    guard = ForecastGuardrail(dlin, xgb_bw, xgb_rpc, pre, thr, cfg.guardrail, sy)

    pred_bw = m_r[:, 0, 0]; true_bw = y_r[:, 0, 0]
    pred_rpc = m_r[:, 0, 1]; true_rpc = y_r[:, 0, 1]
    bb, _, _, _ = run_causal_kf(pred_bw, true_bw,
        sigma_w2=cfg.guardrail.kf_sigma_w2_bw,
        sigma_v2=cfg.guardrail.kf_sigma_v2_bw, P0=cfg.guardrail.kf_p0_bw)
    rb, _, _, _ = run_causal_kf(pred_rpc, true_rpc,
        sigma_w2=cfg.guardrail.kf_sigma_w2_rpc,
        sigma_v2=cfg.guardrail.kf_sigma_v2_rpc, P0=cfg.guardrail.kf_p0_rpc)

    pos, valid = np.searchsorted(
        t_te.values, job_test["t_start"].values.astype("datetime64[ns]"),
        side="right",
    ) - 1, None
    valid = (pos >= 0) & (pos < len(t_te))
    pos_valid = pos[valid]
    jt_valid = job_test.reset_index(drop=True).iloc[valid].reset_index(drop=True)
    if len(pos_valid) == 0:
        return None
    bg_bw = y_r[pos_valid, :, 0].max(1)
    bg_rpc = y_r[pos_valid, :, 1].max(1)
    gt, _ = ground_truth_congestion(
        bg_bw, bg_rpc,
        jt_valid["delta_bw_p90"].values, jt_valid["delta_rpc_p90"].values,
        thr.bw_hard, thr.rpc_hard,
    )
    dec, *_ = guard.evaluate(
        Xte[pos_valid], jt_valid,
        bb[pos_valid], np.zeros(len(pos_valid)),
        rb[pos_valid], np.zeros(len(pos_valid)),
    )
    m = decision_metrics(dec, gt)

    return FoldResult(
        fold=fold,
        train_start=str(t0), train_end=str(tr_end),
        test_start=str(tr_end), test_end=str(te_end),
        rmse_bw=rmse_bw, rmse_rpc=rmse_rpc,
        decision_acc=m["decision_acc"], false_submit_rate=m["false_submit_rate"],
        n_test=len(pos_valid),
        train_sec=time.time() - t0_t,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--train-days", type=int, default=7)
    ap.add_argument("--test-hours", type=int, default=6)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    cfg.data_dir = args.data_dir
    sys_df, job_df, _ = load_all(cfg)

    out = Path(args.report_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold, t0, tr_end, te_end in fold_iter(sys_df, cfg, args.train_days, args.test_hours):
        r = run_fold(fold, t0, tr_end, te_end, sys_df, job_df, cfg)
        if r is None:
            print(f"[fold {fold}] skipped (insufficient data)")
            continue
        print(f"[fold {fold}] acc={r.decision_acc:.3f} fsr={r.false_submit_rate:.3f} "
              f"rmse_bw={r.rmse_bw:.3f} n={r.n_test}")
        rows.append(asdict(r))
    if rows:
        pd.DataFrame(rows).to_csv(out / "rolling_backtest.csv", index=False)
        print(f"[rolling] wrote {out / 'rolling_backtest.csv'}")


if __name__ == "__main__":
    main()
