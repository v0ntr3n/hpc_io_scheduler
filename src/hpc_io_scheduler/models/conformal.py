"""Conformal prediction wrapper for DLinear forecasts (Tier-2 D).

Replaces Gaussian-NLL std with empirical conformal intervals fit on a
calibration set. Distribution-free, sharper coverage than Gaussian.
"""
from __future__ import annotations

import numpy as np


class ConformalForecaster:
    """Conformalize a point predictor (mean) into calibrated intervals.

    Parameters
    ----------
    residuals : (n_calib, S, T) array of point-pred residuals on calibration set
                = y_calib - mean_calib  (both in *real* scale)
    alpha     : 1 - target coverage. Default 0.05 -> 95% interval.
    """

    def __init__(self, residuals: np.ndarray, alpha: float = 0.05):
        self.alpha = float(alpha)
        # Per-horizon, per-target quantile of |residual|
        abs_r = np.abs(residuals)
        S, T = abs_r.shape[1], abs_r.shape[2]
        self.quantile = np.quantile(
            abs_r, 1 - self.alpha, axis=0
        )  # shape (S, T)

    def wrap(self, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) of the conformal band around `mean`."""
        # Broadcast (S,T) over batch
        lo = mean - self.quantile
        hi = mean + self.quantile
        return lo, hi

    def std_proxy(self) -> np.ndarray:
        """Half-width as std proxy for downstream code expecting std."""
        return self.quantile / 1.96


def fit_conformal(
    y_calib: np.ndarray, mean_calib: np.ndarray, alpha: float = 0.05
) -> ConformalForecaster:
    return ConformalForecaster(y_calib - mean_calib, alpha=alpha)
