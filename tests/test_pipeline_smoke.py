"""Smoke test: full pipeline fit on synthetic data, then predict on test split."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpc_io_scheduler.config import Config
from hpc_io_scheduler.data.loader import load_all


def test_load_all_smoke(fake_sys_df, fake_job_df, cfg_min, tmp_path):
    (tmp_path / "system_state_5min.csv").write_text(fake_sys_df.to_csv(index=False))
    (tmp_path / "job_impact_dataset.csv").write_text(fake_job_df.to_csv(index=False))
    cfg = Config()
    cfg.data_dir = str(tmp_path)
    sys_df, job_df, node_df = load_all(cfg)
    assert len(sys_df) == 200
    assert len(job_df) == 50
    assert node_df is None  # node-data.csv not present


def test_windowing_shapes(fake_sys_df, cfg_min, tmp_path):
    from hpc_io_scheduler.data.windowing import fit_system_scalers, make_windows

    cutoff = fake_sys_df["bin"].iloc[100]
    sx, sy = fit_system_scalers(fake_sys_df, cutoff)
    X, Y, t = make_windows(fake_sys_df, sx, sy, cfg_min)
    L, S = cfg_min.data.past_bins, cfg_min.data.future_bins
    assert X.shape[1:] == (L, 7)
    assert Y.shape[1:] == (S, 2)
    assert len(t) == len(X)
