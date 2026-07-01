"""Conformal predictor: empirical coverage on test ≈ 1 - alpha."""
from __future__ import annotations

import numpy as np
import pytest

from hpc_io_scheduler.models.conformal import ConformalForecaster, fit_conformal


def test_conformal_coverage_close_to_target():
    rng = np.random.default_rng(0)
    n_calib, n_test, S, T = 500, 2000, 6, 2
    # True error distribution: N(0, 0.1)
    true_calib = rng.normal(0, 0.1, (n_calib, S, T))
    cf = fit_conformal(true_calib, np.zeros_like(true_calib), alpha=0.05)
    # Test points: mean=0, residuals from same distribution
    mean = np.zeros((n_test, S, T))
    test_resid = rng.normal(0, 0.1, (n_test, S, T))
    lo, hi = cf.wrap(mean + test_resid)  # wrap actual = mean + resid
    coverage = ((test_resid >= -cf.quantile) & (test_resid <= cf.quantile)).mean()
    # Should be >= 1 - alpha (conformal guarantee)
    assert coverage >= 0.94, f"coverage {coverage:.3f} below 1-alpha=0.95"


def test_conformal_quantile_shape():
    cf = ConformalForecaster(np.random.default_rng(0).normal(0, 1, (100, 6, 2)))
    assert cf.quantile.shape == (6, 2)


def test_conformal_quantile_monotone_in_alpha():
    r = np.random.default_rng(0).normal(0, 1, (100, 4, 2))
    q_lo = ConformalForecaster(r, alpha=0.10).quantile
    q_hi = ConformalForecaster(r, alpha=0.01).quantile
    assert (q_hi >= q_lo).all()
