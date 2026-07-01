"""Baselines: FCFS, forecast-point, forecast-prob, LLM-only, FIAC.

CRITICAL: all baselines use PREDICTED delta-IO, never ground-truth.
GT compute uses actual peak for congestion label only (target).
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb


def fcfs(n: int) -> np.ndarray:
    return np.array(["SUBMIT"] * n)


def forecast_point(
    pred_bw_max: np.ndarray,
    pred_rpc_max: np.ndarray,
    delta_bw_pred: np.ndarray,
    delta_rpc_pred: np.ndarray,
    bw_soft: float, bw_hard: float,
    rpc_soft: float, rpc_hard: float,
) -> np.ndarray:
    io_bw = pred_bw_max + delta_bw_pred
    io_rpc = pred_rpc_max + delta_rpc_pred
    submit = (io_bw <= bw_soft) & (io_rpc <= rpc_soft)
    hold = (io_bw > bw_hard) | (io_rpc > rpc_hard)
    return np.where(submit, "SUBMIT", np.where(hold, "HOLD", "THROTTLE"))


def forecast_prob(
    predU_bw_max: np.ndarray,
    predU_rpc_max: np.ndarray,
    delta_bw_pred: np.ndarray,
    delta_rpc_pred: np.ndarray,
    bw_soft: float, bw_hard: float,
    rpc_soft: float, rpc_hard: float,
) -> np.ndarray:
    io_bw = predU_bw_max + delta_bw_pred
    io_rpc = predU_rpc_max + delta_rpc_pred
    submit = (io_bw <= bw_soft) & (io_rpc <= rpc_soft)
    hold = (io_bw > bw_hard) | (io_rpc > rpc_hard)
    return np.where(submit, "SUBMIT", np.where(hold, "HOLD", "THROTTLE"))


def ground_truth_congestion(
    bg_bw_max: np.ndarray, bg_rpc_max: np.ndarray,
    job_dbw: np.ndarray, job_drpc: np.ndarray,
    bw_hard: float, rpc_hard: float,
    use_hard: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """GT label uses ACTUAL delta-IO (oracle). Only used for evaluation."""
    bw_ceil = bw_hard if use_hard else None
    rpc_ceil = rpc_hard if use_hard else None
    actual_bw = bg_bw_max + job_dbw
    actual_rpc = bg_rpc_max + job_drpc
    cong = (actual_bw > bw_ceil) | (actual_rpc > rpc_ceil)
    dec = np.where(cong, "HOLD", "SUBMIT")
    return cong, dec
