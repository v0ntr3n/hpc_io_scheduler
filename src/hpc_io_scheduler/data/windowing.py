"""Sliding-window feature/target tensors from system timeseries."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from hpc_io_scheduler.config import Config
from hpc_io_scheduler.data.loader import SYS_FEATURES, SYS_TARGETS, load_system


def fit_system_scalers(
    sys_df: pd.DataFrame, cutoff: pd.Timestamp
) -> tuple[StandardScaler, StandardScaler]:
    """Scalers fit on train (bins <= cutoff) only. No leakage."""
    mask = (sys_df["bin"] <= cutoff).values
    sx = StandardScaler().fit(sys_df.loc[mask, SYS_FEATURES])
    sy = StandardScaler().fit(sys_df.loc[mask, SYS_TARGETS])
    return sx, sy


def make_windows(
    sys_df: pd.DataFrame,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Sliding windows: X=(N, past, F), Y=(N, future, T)."""
    X_raw = scaler_x.transform(sys_df[SYS_FEATURES]).astype(np.float32)
    Y_raw = scaler_y.transform(sys_df[SYS_TARGETS]).astype(np.float32)

    N = len(sys_df)
    L, S = cfg.data.past_bins, cfg.data.future_bins
    n = N - L - S + 1
    if n <= 0:
        raise ValueError(f"system timeseries too short: N={N}, L+S={L+S}")

    X = np.empty((n, L, X_raw.shape[1]), dtype=np.float32)
    Y = np.empty((n, S, Y_raw.shape[1]), dtype=np.float32)
    t = np.empty(n, dtype="datetime64[ns]")

    # Vectorized sliding view: avoids Python loop.
    idx = np.arange(N)
    for i in range(n):
        sl = slice(i, i + L)
        X[i] = X_raw[sl]
        Y[i] = Y_raw[i + L : i + L + S]
        t[i] = sys_df["bin"].iloc[i + L]

    return X, Y, pd.Series(t, name="t_pred")


def split_windows_by_horizon(
    X: np.ndarray,
    Y: np.ndarray,
    t_pred: pd.Series,
    cutoff: pd.Timestamp,
    horizon_min: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    end = t_pred + pd.Timedelta(minutes=horizon_min)
    mask = (end <= cutoff).values
    return X[mask], Y[mask], X[~mask], Y[~mask], t_pred[~mask].reset_index(drop=True)
