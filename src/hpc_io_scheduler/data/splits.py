"""Leakage-safe splits + utilities for time-aware CV."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hpc_io_scheduler.data.loader import JOB_CAT, JOB_NUM

REQUIRED_FOLD_REFITS: Final = (
    "scaler_x",
    "scaler_y",
    "thresholds",
    "conformal",
    "job_preprocessor",
    "dlinear",
    "xgb_bw",
    "xgb_rpc",
)


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    train_window: pd.Timedelta
    test_window: pd.Timedelta
    stride: pd.Timedelta


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def cutoff_from_jobs(job_df: pd.DataFrame) -> pd.Timestamp:
    """Train cutoff = last train-split job start time."""
    if "split" not in job_df.columns:
        raise ValueError("job_df must contain 'split' column")
    train = job_df[job_df["split"] == "train"]
    if len(train) == 0:
        raise ValueError("no train jobs; cannot derive cutoff")
    return train["t_start"].max()


def build_job_preprocessor() -> ColumnTransformer:
    """OneHot for dnn_label, standardize numeric features."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), JOB_NUM),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                JOB_CAT,
            ),
        ]
    )


def fit_transform_jobs(
    job_train: pd.DataFrame, job_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    pre = build_job_preprocessor()
    X_train = np.asarray(pre.fit_transform(job_train)).astype(np.float32)
    X_test = np.asarray(pre.transform(job_test)).astype(np.float32)
    return X_train, X_test, pre


def group_kfold_indices(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Wrap sklearn GroupKFold to return list of (train_idx, val_idx)."""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(groups)), groups=groups))


def time_series_train_test(
    df: pd.DataFrame, time_col: str, cutoff: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple chronological split. Use when no per-row 'split' column exists."""
    train = df[df[time_col] <= cutoff].reset_index(drop=True)
    test = df[df[time_col] > cutoff].reset_index(drop=True)
    return train, test


def build_walk_forward_folds(
    times: pd.Series,
    cfg: WalkForwardConfig,
) -> list[WalkForwardFold]:
    if cfg.train_window <= pd.Timedelta(0) or cfg.test_window <= pd.Timedelta(0):
        raise ValueError("train_window and test_window must be positive")
    if cfg.stride <= pd.Timedelta(0):
        raise ValueError("stride must be positive")
    ordered = pd.to_datetime(times).sort_values().reset_index(drop=True)
    if ordered.empty:
        return []
    start = pd.Timestamp(ordered.iloc[0])
    last = pd.Timestamp(ordered.iloc[-1])
    folds: list[WalkForwardFold] = []
    fold = 0
    while start + cfg.train_window + cfg.test_window <= last:
        train_end = start + cfg.train_window
        test_end = train_end + cfg.test_window
        folds.append(
            WalkForwardFold(
                fold=fold,
                train_start=start,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
            )
        )
        start += cfg.stride
        fold += 1
    return folds


def assign_walk_forward_split(
    df: pd.DataFrame,
    time_col: str,
    fold: WalkForwardFold,
) -> pd.DataFrame:
    times = pd.to_datetime(df[time_col])
    mask = (times >= fold.train_start) & (times < fold.test_end)
    out = df.loc[mask].copy().reset_index(drop=True)
    out["split"] = np.where(out[time_col] < fold.test_start, "train", "test")
    return out


def align_jobs_to_windows(
    job_starts: np.ndarray, window_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each job_start, find latest window whose t_pred <= job_start.

    Returns (pos, valid_mask) where pos is the window index, valid_mask=False
    for jobs that fall before the first window.
    """
    pos = np.searchsorted(window_times, job_starts, side="right") - 1
    valid = (pos >= 0) & (pos < len(window_times))
    return pos, valid
