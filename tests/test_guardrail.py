"""Guardrail logic: SUBMIT/THROTTLE/HOLD boundaries + authority override."""
from __future__ import annotations

import numpy as np
import pytest

from hpc_io_scheduler.guardrail.thresholds import Thresholds


def make_guard_dummy(thr: Thresholds):
    """Skip ForecastGuardrail wiring; test Thresholds math + override logic."""
    from hpc_io_scheduler.guardrail.policy import ForecastGuardrail

    class _Stub:
        bw_soft, bw_hard = thr.bw_soft, thr.bw_hard
        rpc_soft, rpc_hard = thr.rpc_soft, thr.rpc_hard

    g = _Stub()
    g.bw_soft, g.bw_hard = thr.bw_soft, thr.bw_hard
    g.rpc_soft, g.rpc_hard = thr.rpc_soft, thr.rpc_hard
    return g


def test_authority_override_forces_hold_on_hard_violation():
    thr = Thresholds(bw_soft=1.0, bw_hard=2.0, rpc_soft=1.0, rpc_hard=2.0)
    dec = np.array(["SUBMIT", "THROTTLE", "HOLD", "SUBMIT"])
    bw = np.array([1.0, 1.5, 1.0, 3.0])
    rpc = np.array([1.0, 1.5, 1.0, 1.0])

    from hpc_io_scheduler.guardrail.policy import ForecastGuardrail
    out = ForecastGuardrail.authority_override(dec, bw, rpc, thr.bw_hard, thr.rpc_hard)
    assert out[0] == "SUBMIT"
    assert out[1] == "THROTTLE"
    assert out[2] == "HOLD"
    assert out[3] == "HOLD"


def test_thresholds_quantile_train_only(tmp_path):
    import pandas as pd
    from hpc_io_scheduler.guardrail.thresholds import compute_thresholds
    from hpc_io_scheduler.config import GuardrailConfig

    t = pd.date_range("2021-01-01", periods=200, freq="5min")
    df = pd.DataFrame({
        "bin": t,
        "bw_recon_mbps": np.linspace(0.0, 1.0, 200),
        "lustre_rpc": np.linspace(1e6, 1e8, 200),
    })
    cutoff = t[100]
    thr = compute_thresholds(df, cutoff, GuardrailConfig())
    # Quantiles of first half only
    assert thr.bw_soft == pytest.approx(0.5 * 0.95, rel=0.05)
    assert thr.bw_hard >= thr.bw_soft
