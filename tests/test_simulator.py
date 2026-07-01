"""Closed-loop simulator determinism + non-negative metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpc_io_scheduler.config import SimConfig
from hpc_io_scheduler.evaluation.simulator import simulate


def _build_jobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2021-06-01")
    return pd.DataFrame({
        "id_job_norm": [f"u_{i:04d}" for i in range(n)],
        "t_start": pd.date_range(t0, periods=n, freq="15min"),
        "delta_bw_p90": rng.normal(0.01, 0.005, n),
        "delta_rpc_p90": rng.normal(1e5, 1e4, n),
    })


def test_simulator_deterministic_same_seed():
    jobs = _build_jobs(100)
    sys = pd.DataFrame({
        "bin": pd.date_range("2021-06-01", periods=200, freq="5min"),
        "active_jobs": np.full(200, 30),
    })
    dec = np.where(np.arange(100) % 2 == 0, "SUBMIT", "HOLD")
    cong = np.zeros(100, dtype=bool)
    cfg = SimConfig()
    a = simulate(jobs, dec, cong, sys, None, cfg)
    b = simulate(jobs, dec, cong, sys, None, cfg)
    assert a["completed_jobs"] == b["completed_jobs"]
    assert a["throughput_jobs_hr"] == pytest.approx(b["throughput_jobs_hr"])


def test_simulator_metrics_nonnegative_and_within_bounds():
    jobs = _build_jobs(50)
    sys = pd.DataFrame({
        "bin": pd.date_range("2021-06-01", periods=200, freq="5min"),
        "active_jobs": np.full(200, 20),
    })
    dec = np.array(["SUBMIT"] * 50)
    cong = np.zeros(50, dtype=bool)
    cfg = SimConfig()
    m = simulate(jobs, dec, cong, sys, None, cfg)
    assert m["completed_jobs"] >= 0
    assert 0.0 <= m["utilisation_pct"] <= 100.0
    assert 0.0 <= m["congestion_rate_pct"] <= 100.0
    assert m["throughput_jobs_hr"] >= 0.0
    assert m["avg_wait_sec"] >= 0.0
    assert m["avg_queue_len"] >= 0.0


def test_simulator_capacity_bounds_concurrency():
    jobs = _build_jobs(500)
    sys = pd.DataFrame({
        "bin": pd.date_range("2021-06-01", periods=200, freq="5min"),
        "active_jobs": np.full(200, 25),
    })
    dec = np.array(["SUBMIT"] * 500)
    cong = np.zeros(500, dtype=bool)
    cfg = SimConfig(capacity_quantile=0.95, capacity_min=50)
    m = simulate(jobs, dec, cong, sys, None, cfg)
    # capacity = max(quantile, min) = max(25, 50) = 50
    assert m["capacity"] == 50
