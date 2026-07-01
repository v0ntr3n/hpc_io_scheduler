"""Forecast guardrail policy: SUBMIT / THROTTLE / HOLD with safety override.

Inputs: per-window upper bounds (BW, RPC) + per-job delta-IO.
Authority: physics guardrail > learned risk > LLM advisor.

Features
- Conformal std (when fitted) replaces Gaussian std.
- Adaptive thresholds shift by recent KF bias (when auto_tune=True).
- Cost-sensitive submit allows high-priority jobs through softer bounds.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
import torch

from hpc_io_scheduler.config import GuardrailConfig
from hpc_io_scheduler.guardrail.thresholds import Thresholds, adaptive_thresholds


class ForecastGuardrail:
    def __init__(
        self,
        forecaster: torch.nn.Module,
        xgb_bw: xgb.Booster,
        xgb_rpc: xgb.Booster,
        preprocessor,
        thresholds: Thresholds,
        cfg: GuardrailConfig,
        scaler_y,
        device: str | torch.device = "cpu",
        conformal=None,
    ):
        self.fmodel = forecaster
        self.xgb_bw = xgb_bw
        self.xgb_rpc = xgb_rpc
        self.pre = preprocessor
        self.thr = thresholds
        self.cfg = cfg
        self.scaler_y = scaler_y
        self.device = device
        self.sustained = cfg.sustained_bins
        self.conformal = conformal

    @torch.no_grad()
    def _forecast(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.fmodel.eval().to(self.device)
        mean, std = self.fmodel(torch.tensor(windows, dtype=torch.float32, device=self.device))
        mean = mean.cpu().numpy()
        std = std.cpu().numpy()
        shp = mean.shape
        mean_r = self.scaler_y.inverse_transform(mean.reshape(-1, 2)).reshape(shp)
        std_r = std * self.scaler_y.scale_
        if self.conformal is not None:
            std_r = np.broadcast_to(
                self.conformal.std_proxy(), std_r.shape
            ).copy()
        return mean_r, std_r

    def evaluate(
        self,
        windows: np.ndarray,
        job_df,
        bias_bw: np.ndarray,
        P_bw: np.ndarray,
        bias_rpc: np.ndarray,
        P_rpc: np.ndarray,
        priorities: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mean_r, std_r = self._forecast(windows)
        z = self.cfg.z

        thr = self.thr
        if self.cfg.auto_tune:
            thr = adaptive_thresholds(
                thr,
                bias_bw=float(np.mean(bias_bw)),
                bias_rpc=float(np.mean(bias_rpc)),
                alpha=0.3,
            )

        mu_bw = mean_r[:, :, 0] + bias_bw[:, None]
        mu_rpc = mean_r[:, :, 1] + bias_rpc[:, None]
        sg_bw = np.sqrt(std_r[:, :, 0] ** 2 + P_bw[:, None])
        sg_rpc = np.sqrt(std_r[:, :, 1] ** 2 + P_rpc[:, None])

        upper_bw = mu_bw + z * sg_bw
        upper_rpc = mu_rpc + z * sg_rpc

        hold_bw = (upper_bw > thr.bw_hard).sum(1) >= self.sustained
        hold_rpc = (upper_rpc > thr.rpc_hard).sum(1) >= self.sustained

        bw_fc = np.percentile(upper_bw, 75, axis=1)
        rpc_fc = np.percentile(upper_rpc, 75, axis=1)

        Xj = np.asarray(self.pre.transform(job_df)).astype(np.float32)
        dmat = xgb.DMatrix(Xj)
        bw_bound = bw_fc + self.xgb_bw.predict(dmat)
        rpc_bound = rpc_fc + self.xgb_rpc.predict(dmat)

        submit = (bw_bound <= thr.bw_soft) & (rpc_bound <= thr.rpc_soft)
        if priorities is not None:
            prio = np.asarray(priorities, dtype=float)
            high = prio >= 90
            low_util = (bw_bound < thr.bw_soft * 0.8) & (rpc_bound < thr.rpc_soft * 0.8)
            submit = submit & (high | low_util)

        hold = (
            hold_bw
            | hold_rpc
            | (bw_bound > thr.bw_hard * self.cfg.hard_buffer)
            | (rpc_bound > thr.rpc_hard * self.cfg.hard_buffer)
        )
        gray = ~(submit | hold)
        dec = np.where(submit, "SUBMIT", np.where(hold, "HOLD", "THROTTLE"))
        return dec, bw_bound, rpc_bound, gray

    @staticmethod
    def authority_override(dec: np.ndarray, bw_bound: np.ndarray, rpc_bound: np.ndarray,
                           bw_hard: float, rpc_hard: float) -> np.ndarray:
        out = dec.copy()
        mask = (bw_bound > bw_hard) | (rpc_bound > rpc_hard)
        out[mask] = "HOLD"
        return out
