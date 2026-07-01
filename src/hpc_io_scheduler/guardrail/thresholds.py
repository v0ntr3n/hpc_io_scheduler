"""Threshold computation: empirical quantiles, train-only (no leakage)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hpc_io_scheduler.config import GuardrailConfig
from hpc_io_scheduler.data.loader import SYS_TARGETS


@dataclass
class Thresholds:
    bw_soft: float
    bw_hard: float
    rpc_soft: float
    rpc_hard: float
    mode: str = "empirical_quantile_train_only"

    def to_dict(self) -> dict:
        return {
            "bw_threshold_soft": self.bw_soft,
            "bw_threshold_hard": self.bw_hard,
            "rpc_threshold_soft": self.rpc_soft,
            "rpc_threshold_hard": self.rpc_hard,
            "threshold_mode": self.mode,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Thresholds":
        d = json.load(open(path))
        return cls(
            bw_soft=d["bw_threshold_soft"],
            bw_hard=d["bw_threshold_hard"],
            rpc_soft=d["rpc_threshold_soft"],
            rpc_hard=d["rpc_threshold_hard"],
            mode=d.get("threshold_mode", "unknown"),
        )


def compute_thresholds(
    sys_df: pd.DataFrame, cutoff: pd.Timestamp, cfg: GuardrailConfig
) -> Thresholds:
    """Quantile-based thresholds, fit on train bins (bin <= cutoff) only."""
    train = sys_df["bin"] <= cutoff
    bw = sys_df.loc[train, SYS_TARGETS[0]]
    rpc = sys_df.loc[train, SYS_TARGETS[1]]
    return Thresholds(
        bw_soft=float(bw.quantile(cfg.bw_quantile_soft)),
        bw_hard=float(bw.quantile(cfg.bw_quantile_hard)),
        rpc_soft=float(rpc.quantile(cfg.rpc_quantile_soft)),
        rpc_hard=float(rpc.quantile(cfg.rpc_quantile_hard)),
    )


def adaptive_thresholds(
    base: Thresholds, bias_bw: float, bias_rpc: float, alpha: float = 0.3
) -> Thresholds:
    """Tier-3 H: shift thresholds by recent KF bias."""
    return Thresholds(
        bw_soft=base.bw_soft + alpha * bias_bw,
        bw_hard=base.bw_hard + alpha * bias_bw,
        rpc_soft=base.rpc_soft + alpha * bias_rpc,
        rpc_hard=base.rpc_hard + alpha * bias_rpc,
        mode=f"adaptive(alpha={alpha})",
    )
