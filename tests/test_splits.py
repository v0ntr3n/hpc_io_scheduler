"""Splits must be chronological and GroupKFold must keep groups disjoint."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpc_io_scheduler.data.splits import (
    REQUIRED_FOLD_REFITS,
    WalkForwardConfig,
    align_jobs_to_windows,
    assign_walk_forward_split,
    build_walk_forward_folds,
    cutoff_from_jobs,
    group_kfold_indices,
    time_series_train_test,
)


def test_cutoff_from_jobs_returns_max_train_time(fake_job_df: pd.DataFrame):
    cutoff = cutoff_from_jobs(fake_job_df)
    train = fake_job_df[fake_job_df["split"] == "train"]
    assert cutoff == train["t_start"].max()


def test_time_series_train_test_no_overlap(fake_job_df: pd.DataFrame):
    cutoff = cutoff_from_jobs(fake_job_df)
    tr, te = time_series_train_test(fake_job_df, "t_start", cutoff)
    assert (tr["t_start"] <= cutoff).all()
    assert (te["t_start"] > cutoff).all()


def test_group_kfold_disjoint_groups():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 5, 100)
    for tr, va in group_kfold_indices(groups, n_splits=3):
        assert set(groups[tr]).isdisjoint(set(groups[va]))


def test_align_jobs_to_windows_handles_edge_cases():
    windows = pd.date_range("2021-06-01", periods=10, freq="5min").values
    jobs_before = np.array([windows[0] - np.timedelta64(1, "m")], dtype="datetime64[ns]")
    jobs_in = np.array([windows[3] + np.timedelta64(1, "m")], dtype="datetime64[ns]")
    pos, valid = align_jobs_to_windows(
        np.concatenate([jobs_before, jobs_in]), windows,
    )
    assert not valid[0]   # before first window
    assert valid[1] and pos[1] == 3


def test_walk_forward_folds_are_chronological():
    times = pd.date_range("2021-06-01", periods=10 * 24, freq="1h")
    cfg = WalkForwardConfig(
        train_window=pd.Timedelta(days=2),
        test_window=pd.Timedelta(hours=12),
        stride=pd.Timedelta(hours=12),
    )

    folds = build_walk_forward_folds(pd.Series(times), cfg)

    assert folds
    assert folds[0].train_start == times[0]
    assert folds[0].train_end == folds[0].test_start
    assert folds[0].test_end <= times[-1]
    assert folds[1].train_start == folds[0].train_start + cfg.stride


def test_walk_forward_folds_reject_nonpositive_stride():
    times = pd.date_range("2021-06-01", periods=24, freq="1h")
    cfg = WalkForwardConfig(
        train_window=pd.Timedelta(hours=4),
        test_window=pd.Timedelta(hours=2),
        stride=pd.Timedelta(0),
    )

    with pytest.raises(ValueError, match="stride"):
        build_walk_forward_folds(pd.Series(times), cfg)


def test_assign_walk_forward_split_uses_only_fold_window(fake_job_df: pd.DataFrame):
    fold = build_walk_forward_folds(
        fake_job_df["t_start"],
        WalkForwardConfig(
            train_window=pd.Timedelta(hours=10),
            test_window=pd.Timedelta(hours=5),
            stride=pd.Timedelta(hours=5),
        ),
    )[0]

    split = assign_walk_forward_split(fake_job_df, "t_start", fold)

    assert set(split["split"]) == {"train", "test"}
    assert split[split["split"] == "train"]["t_start"].max() < fold.train_end
    assert split[split["split"] == "test"]["t_start"].min() >= fold.test_start
    assert split["t_start"].max() < fold.test_end


def test_required_fold_refits_names_all_learned_components():
    assert REQUIRED_FOLD_REFITS == (
        "scaler_x",
        "scaler_y",
        "thresholds",
        "conformal",
        "job_preprocessor",
        "dlinear",
        "xgb_bw",
        "xgb_rpc",
    )
