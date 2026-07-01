"""Shared fixtures for smoke tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_sys_df() -> pd.DataFrame:
    n = 200
    t = pd.date_range("2021-06-01", periods=n, freq="5min")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "bin": t,
            "bw_recon_mbps": rng.normal(0.2, 0.05, n).clip(0),
            "io_p95": rng.normal(0.1, 0.02, n),
            "io_max": rng.normal(0.3, 0.05, n),
            "io_fano": rng.normal(0.5, 0.1, n),
            "lustre_rpc": rng.normal(2.7e7, 1e6, n),
            "active_jobs": rng.integers(10, 50, n),
            "sum_file_rw": rng.normal(1e9, 1e8, n),
        }
    )


@pytest.fixture
def fake_job_df(fake_sys_df: pd.DataFrame) -> pd.DataFrame:
    n = 50
    t0 = fake_sys_df["bin"].iloc[24]  # 2h in
    t = pd.date_range(t0, periods=n, freq="30min")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "id_job_norm": [f"u_{i:04d}_{i:04d}" for i in range(n)],
            "t_start": t,
            "split": ["train"] * (n - 10) + ["test"] * 10,
            "dnn_label": rng.choice(["training", "unlabeled"], n),
            "mem_req": rng.integers(1, 64, n),
            "cpus_req": rng.integers(1, 32, n),
            "nodes_alloc": rng.integers(1, 4, n),
            "gres_alloc": rng.integers(0, 2, n),
            "hist_io_mean": rng.normal(0.1, 0.02, n),
            "hist_startup": rng.normal(60, 10, n),
            "hist_ckpt": rng.normal(5, 1, n),
            "hist_term": rng.normal(0.5, 0.1, n),
            "hist_burst": rng.normal(0.2, 0.05, n),
            "delta_bw_p90": rng.normal(0.01, 0.005, n),
            "delta_rpc_p90": rng.normal(1e5, 1e4, n),
        }
    )


@pytest.fixture
def cfg_min(tmp_path):
    from hpc_io_scheduler.config import Config

    c = Config()
    c.data_dir = str(tmp_path)
    (tmp_path).mkdir(exist_ok=True)
    return c
