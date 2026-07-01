"""Online causal Kalman residual correction.

CRITICAL: prior-based. The bias used to correct prediction at step i is
the prior accumulated from steps 0..i-1 only (no leakage).
"""
from __future__ import annotations

import numpy as np


class KalmanResidual:
    def __init__(self, sigma_w2: float = 1e-4, sigma_v2: float = 1e-2, P0: float = 1.0):
        self.sw = float(sigma_w2)
        self.sv = float(sigma_v2)
        self.r = 0.0
        self.P = float(P0)
        self.last_nis = np.nan

    def update(self, residual: float) -> tuple[float, float, float]:
        self.P = self.P + self.sw
        S = self.P + self.sv
        K = self.P / S
        nu = residual - self.r
        self.r = self.r + K * nu
        self.P = (1.0 - K) * self.P
        self.last_nis = (nu * nu) / S
        return self.r, self.P, self.last_nis


def run_causal_kf(
    pred_seq: np.ndarray, true_seq: np.ndarray, **kw
) -> tuple[np.ndarray, np.ndarray, np.ndarray, KalmanResidual]:
    """Strictly causal: prior at step i uses only info from 0..i-1.

    Returns (bias_prior, P_prior, nis, final_kf_state).
    """
    kf = KalmanResidual(**kw)
    n = len(pred_seq)
    bias = np.zeros(n)
    P = np.zeros(n)
    nis = np.zeros(n)
    for i in range(n):
        # Lock prior (does not look at true_seq[i])
        bias[i] = kf.r
        P[i] = kf.P
        # Observe, then update posterior
        residual = true_seq[i] - pred_seq[i]
        _, _, d = kf.update(residual)
        nis[i] = d
    return bias, P, nis, kf
