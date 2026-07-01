"""End-to-end pipeline orchestration. One function = one full run."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from hpc_io_scheduler.config import Config, ARTIFACT_DIR, REPORT_DIR
from hpc_io_scheduler.data.loader import load_all
from hpc_io_scheduler.data.splits import (
    cutoff_from_jobs,
    fit_transform_jobs,
    align_jobs_to_windows,
)
from hpc_io_scheduler.data.windowing import (
    fit_system_scalers,
    make_windows,
    split_windows_by_horizon,
)
from hpc_io_scheduler.guardrail.kalman import run_causal_kf
from hpc_io_scheduler.guardrail.policy import ForecastGuardrail
from hpc_io_scheduler.guardrail.thresholds import compute_thresholds
from hpc_io_scheduler.models.conformal import fit_conformal
from hpc_io_scheduler.models.dlinear import train_dlinear, predict_dlinear
from hpc_io_scheduler.models.xgb_delta import timed_train as train_xgb_timed, predict_xgb, weighted_rmse


@dataclass
class FittedBundle:
    cfg: Config
    cutoff: pd.Timestamp
    scaler_x: object
    scaler_y: object
    pre: object
    thresholds: object
    forecaster: torch.nn.Module
    forecaster_kind: str
    conformal: object
    xgb_bw: object
    xgb_rpc: object
    job_groups: np.ndarray
    job_weights: np.ndarray


def fit_all(cfg: Config) -> FittedBundle:
    sys_df, job_df, _ = load_all(cfg)
    cutoff = cutoff_from_jobs(job_df)

    sx, sy = fit_system_scalers(sys_df, cutoff)
    X, Y, t_pred = make_windows(sys_df, sx, sy, cfg)
    Xtr, Ytr, Xte, Yte, t_te = split_windows_by_horizon(
        X, Y, t_pred, cutoff, horizon_min=5 * cfg.data.future_bins
    )

    if cfg.model.forecaster == "nbeats":
        from hpc_io_scheduler.models.nbeats import train_nbeats, predict_nbeats
        forecaster, _ = train_nbeats(Xtr, Ytr, cfg, device=cfg_device(),
                                     n_blocks=cfg.model.nbeats_blocks,
                                     hidden=cfg.model.nbeats_hidden)
        predict_fn = predict_nbeats
        kind = "nbeats"
    else:
        forecaster, _ = train_dlinear(Xtr, Ytr, cfg, device=cfg_device())
        predict_fn = predict_dlinear
        kind = "dlinear"

    n_calib = min(1000, len(Xtr) // 5)
    Xcal, Ycal = Xtr[-n_calib:], Ytr[-n_calib:]
    m_cal, _ = predict_fn(forecaster, Xcal, device=cfg_device())
    shp = m_cal.shape
    m_cal_r = sy.inverse_transform(m_cal.reshape(-1, 2)).reshape(shp)
    y_cal_r = sy.inverse_transform(Ycal.reshape(-1, 2)).reshape(shp)
    conformal = fit_conformal(y_cal_r, m_cal_r, alpha=0.05)

    thr = compute_thresholds(sys_df, cutoff, cfg.guardrail)

    job_train = job_df[job_df["split"] == "train"].reset_index(drop=True)
    job_test = job_df[job_df["split"] == "test"].reset_index(drop=True)
    Xj_tr, Xj_te, pre = fit_transform_jobs(job_train, job_test)
    groups = job_train["id_job_norm"].str.split("_").str[0].values
    w_tr = np.where(job_train["dnn_label"] != "unlabeled", 5.0, 1.0)
    w_te = np.where(job_test["dnn_label"] != "unlabeled", 5.0, 1.0)

    yj_bw = job_train["delta_bw_p90"].values
    yj_rpc = job_train["delta_rpc_p90"].values
    xgb_bw, rounds_bw, _ = train_xgb_timed(Xj_tr, yj_bw, w_tr, groups, cfg)
    xgb_rpc, rounds_rpc, _ = train_xgb_timed(Xj_tr, yj_rpc, w_tr, groups, cfg)

    return FittedBundle(
        cfg=cfg, cutoff=cutoff, scaler_x=sx, scaler_y=sy, pre=pre,
        thresholds=thr, forecaster=forecaster, forecaster_kind=kind,
        conformal=conformal,
        xgb_bw=xgb_bw, xgb_rpc=xgb_rpc,
        job_groups=groups, job_weights=w_tr,
    )


def cfg_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_bundle(b: FittedBundle, root: str | Path = ARTIFACT_DIR) -> Path:
    root = Path(root)
    (root / "models").mkdir(parents=True, exist_ok=True)
    fname = "nbeats_t1.pt" if b.forecaster_kind == "nbeats" else "dlinear_t1.pt"
    torch.save(b.forecaster.state_dict(), root / "models" / fname)
    (root / "models" / "forecaster_kind.txt").write_text(b.forecaster_kind)
    b.xgb_bw.save_model(str(root / "models" / "xgb_bw.json"))
    b.xgb_rpc.save_model(str(root / "models" / "xgb_rpc.json"))
    joblib.dump(b.scaler_x, root / "models" / "scaler_system.joblib")
    joblib.dump(b.scaler_y, root / "models" / "scaler_y.joblib")
    joblib.dump(b.pre, root / "models" / "preprocessor_job.joblib")
    joblib.dump(b.conformal, root / "models" / "conformal.joblib")
    b.thresholds.save(str(root / "guardrail_config.json"))
    return root


# ---------------------- CLI entry points --------------------------------

def main_train() -> None:
    cfg = Config()
    b = fit_all(cfg)
    save_bundle(b)
    print(json.dumps({"cutoff": str(b.cutoff), "saved_to": str(ARTIFACT_DIR)}, indent=2))


def main_backtest() -> None:
    from hpc_io_scheduler.backtest import run_backtest
    cfg = Config()
    run_backtest(
        cfg,
        artifact_dir=cfg.artifact_dir,
        report_dir=cfg.report_dir,
        run_audit=False,
    )


def main_serve() -> None:
    raise NotImplementedError("Wire to scripts/serve.py")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        main_train()
    else:
        print("Usage: python -m hpc_io_scheduler.pipeline [train|backtest|serve]")
