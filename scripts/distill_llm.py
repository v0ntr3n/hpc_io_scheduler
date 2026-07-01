"""Distill gray-zone LLM decisions into a tree model (Tier-2 I).

Runs the (Qwen or GGUF) advisor on every gray-zone case in the test split,
collects (ctx features, action) pairs, and trains a small XGBClassifier.
Saves the model to artifacts/models/distilled_advisor.joblib. At inference
time, `DistilledAdvisor(model_path=...)` is a drop-in replacement for
QwenAdvisor — no GPU needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from hpc_io_scheduler.backtest import _load_bundle, job_train_dnn_labels, order_to_inv
from hpc_io_scheduler.config import Config
from hpc_io_scheduler.data.loader import load_all
from hpc_io_scheduler.data.splits import align_jobs_to_windows, cutoff_from_jobs
from hpc_io_scheduler.data.windowing import (
    make_windows, split_windows_by_horizon,
)
from hpc_io_scheduler.guardrail.policy import ForecastGuardrail
from hpc_io_scheduler.models.lora_advisor import QwenAdvisor, heuristic_advise
from hpc_io_scheduler.models.dlinear import predict_dlinear


ACTION_MAP = {"SUBMIT": 0, "THROTTLE": 1, "HOLD": 2}
INV_ACTION = {v: k for k, v in ACTION_MAP.items()}


def featurize(ctx: dict) -> np.ndarray:
    return np.array([
        int(ctx.get("cpus_req", 0)),
        float(ctx["bw_bound"]),
        float(ctx["rpc_bound"]),
        float(ctx.get("priority", 100)),
        int(ctx.get("task_type", "NA") == "training"),
        float(ctx.get("bw_frac", 0.0)),
        float(ctx.get("rpc_frac", 0.0)),
    ], dtype=np.float32)


def collect_decisions(cfg: Config, artifact_dir: str) -> tuple[np.ndarray, np.ndarray]:
    sys_df, job_df, _ = load_all(cfg)
    cutoff = cutoff_from_jobs(job_df)
    sx, sy, pre, thr, dlin, xgb_bw, xgb_rpc = _load_bundle(cfg, Path(artifact_dir))
    guard = ForecastGuardrail(dlin, xgb_bw, xgb_rpc, pre, thr, cfg.guardrail, sy)
    advisor = QwenAdvisor(cfg)

    X, Y, t_pred = make_windows(sys_df, sx, sy, cfg)
    Xtr, Ytr, Xte, Yte, t_te = split_windows_by_horizon(
        X, Y, t_pred, cutoff, horizon_min=5 * cfg.data.future_bins,
    )
    m, s = predict_dlinear(dlin, Xte)
    shp = m.shape
    m_r = sy.inverse_transform(m.reshape(-1, 2)).reshape(shp)

    job_test = job_df[job_df["split"] == "test"].reset_index(drop=True)
    pos, valid = align_jobs_to_windows(
        job_test["t_start"].values.astype("datetime64[ns]"),
        t_te.values.astype("datetime64[ns]"),
    )
    pos_valid = pos[valid]
    job_test_valid = job_test[valid].reset_index(drop=True)
    prio_arr = pd.to_numeric(
        job_test_valid.get("priority", pd.Series([100] * len(job_test_valid))),
        errors="coerce",
    ).fillna(100).values

    dec, bwb, rpcb, gray = guard.evaluate(
        Xte[pos_valid], job_test_valid,
        np.zeros(len(pos_valid)), np.zeros(len(pos_valid)),
        np.zeros(len(pos_valid)), np.zeros(len(pos_valid)),
        priorities=prio_arr,
    )
    gray_idx = np.where(gray)[0]
    if len(gray_idx) == 0:
        return np.empty((0, 7), dtype=np.float32), np.empty(0, dtype=int)

    ctx_l = [{
        "job_id": str(job_test_valid.iloc[i]["id_job_norm"]),
        "task_type": str(job_test_valid.iloc[i]["dnn_label"]),
        "cpus_req": int(job_test_valid.iloc[i].get("cpus_req", 0)),
        "sys_status": "GRAY",
        "bw_bound": float(bwb[i]), "rpc_bound": float(rpcb[i]),
        "bw_hard": thr.bw_hard, "rpc_hard": thr.rpc_hard,
        "priority": float(prio_arr[i]),
        "bw_frac": float(bwb[i]) / max(float(thr.bw_hard), 1e-9),
        "rpc_frac": float(rpcb[i]) / max(float(thr.rpc_hard), 1e-9),
    } for i in gray_idx]

    res = advisor.advise_batch(ctx_l, batch_size=cfg.llm.batch_size)
    Xd, yd = [], []
    for j, i in enumerate(gray_idx):
        action, _ = res[j]
        if action is None:
            action, _ = heuristic_advise(ctx_l[j])
        Xd.append(featurize(ctx_l[j]))
        yd.append(ACTION_MAP[action])
    return np.vstack(Xd), np.asarray(yd, dtype=int)


def train_distilled(X: np.ndarray, y: np.ndarray, out_path: Path) -> xgb.XGBClassifier:
    if len(y) == 0:
        raise SystemExit("No gray-zone cases; nothing to distill.")
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="multi:softprob", num_class=3, tree_method="hist",
    )
    clf.fit(X, y)
    joblib.dump(clf, out_path)
    return clf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--artifact-dir", default="artifacts")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    out = Path(args.artifact_dir) / "models" / "distilled_advisor.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[distill] collecting gray-zone decisions via LLM advisor...")
    X, y = collect_decisions(cfg, args.artifact_dir)
    print(f"[distill] dataset: {X.shape} | dist: {pd.Series(y).value_counts().to_dict()}")
    clf = train_distilled(X, y, out)
    print(f"[distill] saved to {out}")


if __name__ == "__main__":
    main()
