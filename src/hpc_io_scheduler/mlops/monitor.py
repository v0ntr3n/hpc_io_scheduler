"""MLOps monitor: NIS-based drift detector + retrain trigger."""
from __future__ import annotations

import numpy as np

from hpc_io_scheduler.config import MLOpsConfig
from hpc_io_scheduler.guardrail.kalman import KalmanResidual


def qwen_agent_plan(nis_hist: list[float], k_nis: float) -> dict:
    recent = np.array(nis_hist[-20:])
    if len(recent) == 0:
        return {"confidence": 0.0, "action": "NOOP"}
    conf = float(min(1.0, (recent > k_nis).mean() + 0.5))
    return {"confidence": conf, "action": "RETRAIN" if conf >= 0.5 else "HOLD"}


class MLOpsMonitorKF:
    def __init__(self, kf: KalmanResidual, cfg: MLOpsConfig):
        self.kf = kf
        self.cfg = cfg
        self.nis_hist: list[float] = []
        self.retrain_events = 0
        self.hitl_events = 0

    def step(self, pred: float, true: float) -> dict:
        r, P, nis = self.kf.update(true - pred)
        self.nis_hist.append(float(nis))
        if nis > self.cfg.k_nis:
            plan = qwen_agent_plan(self.nis_hist, self.cfg.k_nis)
            if plan["confidence"] >= self.cfg.conf_min:
                self.retrain_events += 1
                return {"retrain": True, "nis": float(nis), "bias": float(r), "P": float(P)}
            self.hitl_events += 1
            return {"hitl": True, "nis": float(nis), "bias": float(r), "P": float(P)}
        return {"nis": float(nis), "bias": float(r), "P": float(P)}

    def summary(self) -> dict:
        return {
            "retrain_events": self.retrain_events,
            "hitl_events": self.hitl_events,
            "mean_nis": float(np.mean(self.nis_hist)) if self.nis_hist else float("nan"),
        }
