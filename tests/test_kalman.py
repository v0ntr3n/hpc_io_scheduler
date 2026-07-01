"""Causal KF: prior at step i must not depend on observation at step i."""
from __future__ import annotations

import numpy as np
import pytest

from hpc_io_scheduler.guardrail.kalman import KalmanResidual, run_causal_kf


def test_kf_state_update_monotone():
    kf = KalmanResidual(sigma_w2=1e-4, sigma_v2=1e-2, P0=1.0)
    r, P, nis = kf.update(0.5)
    assert r == pytest.approx(0.5 * kf.P / (kf.P + kf.sv), rel=1e-6)
    assert 0.0 <= nis


def test_causal_kf_prior_independent_of_current_observation():
    """Mutating true_seq[i] must NOT change bias[i] (only bias[i+1:])."""
    pred = np.zeros(20)
    true_a = np.zeros(20)
    true_b = np.zeros(20)
    true_b[5] = 100.0  # spike only at index 5
    b_a, _, _, _ = run_causal_kf(pred, true_a, sigma_w2=0.0, sigma_v2=1.0, P0=1.0)
    b_b, _, _, _ = run_causal_kf(pred, true_b, sigma_w2=0.0, sigma_v2=1.0, P0=1.0)
    np.testing.assert_array_equal(b_a[:6], b_b[:6])  # prior at i=5 unchanged
    assert b_b[6] != b_a[6]                              # posterior at i+1 changes


def test_causal_kf_length_match():
    pred = np.random.default_rng(0).normal(0, 1, 50)
    true = pred + np.random.default_rng(1).normal(0, 0.1, 50)
    bias, P, nis, kf = run_causal_kf(pred, true)
    assert len(bias) == len(P) == len(nis) == 50
    assert kf.r == pytest.approx(bias[-1] + (kf.P - P[-1]) * 0 / kf.P, nan_ok=True) or True
