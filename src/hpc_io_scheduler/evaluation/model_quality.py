from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np


class ForecastQualityDict(TypedDict):
    rmse_bw: float
    rmse_rpc: float
    n_eval: int
    n_gray: int


class DecisionQualityDict(TypedDict):
    method: str
    decision_acc: float
    false_submit_rate: float
    false_hold_rate: float
    balanced_acc: float


class AdvisorAblationSummaryDict(TypedDict):
    mode: str
    decisions: list[str]
    changed_from_guardrail: int
    safety_overrides: int


class CalibrationSummaryDict(TypedDict):
    overall_coverage: float
    mean_interval_width: float
    by_target: dict[str, float]
    by_horizon: dict[str, float]


class AdvisorDict(TypedDict):
    mode: str
    ablations: list[AdvisorAblationSummaryDict]


class LeakageDict(TypedDict):
    verdict: str
    group_overlap: int
    top_suspicious: list[str]
    hard_fail: bool


class ModelQualityReportDict(TypedDict):
    forecast: ForecastQualityDict
    decisions: list[DecisionQualityDict]
    calibration: CalibrationSummaryDict
    advisor: AdvisorDict
    leakage: LeakageDict


@dataclass(frozen=True, slots=True)
class ForecastQuality:
    rmse_bw: float
    rmse_rpc: float
    n_eval: int
    n_gray: int

    def to_dict(self) -> ForecastQualityDict:
        return {
            "rmse_bw": float(self.rmse_bw),
            "rmse_rpc": float(self.rmse_rpc),
            "n_eval": int(self.n_eval),
            "n_gray": int(self.n_gray),
        }


@dataclass(frozen=True, slots=True)
class DecisionQuality:
    method: str
    decision_acc: float
    false_submit_rate: float
    false_hold_rate: float
    balanced_acc: float

    def to_dict(self) -> DecisionQualityDict:
        return {
            "method": self.method,
            "decision_acc": float(self.decision_acc),
            "false_submit_rate": float(self.false_submit_rate),
            "false_hold_rate": float(self.false_hold_rate),
            "balanced_acc": float(self.balanced_acc),
        }


@dataclass(frozen=True, slots=True)
class AdvisorAblationInput:
    mode: str
    guardrail_decisions: Sequence[str]
    proposed_decisions: Sequence[str]
    bw_bound: np.ndarray
    rpc_bound: np.ndarray
    bw_hard: float
    rpc_hard: float


@dataclass(frozen=True, slots=True)
class AdvisorAblationSummary:
    mode: str
    decisions: Sequence[str]
    changed_from_guardrail: int
    safety_overrides: int

    def to_dict(self) -> AdvisorAblationSummaryDict:
        return {
            "mode": self.mode,
            "decisions": list(self.decisions),
            "changed_from_guardrail": int(self.changed_from_guardrail),
            "safety_overrides": int(self.safety_overrides),
        }


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    overall_coverage: float
    mean_interval_width: float
    by_target: Mapping[str, float]
    by_horizon: Mapping[str, float]

    def to_dict(self) -> CalibrationSummaryDict:
        return {
            "overall_coverage": float(self.overall_coverage),
            "mean_interval_width": float(self.mean_interval_width),
            "by_target": {name: float(value) for name, value in self.by_target.items()},
            "by_horizon": {name: float(value) for name, value in self.by_horizon.items()},
        }


@dataclass(frozen=True, slots=True)
class LeakageSummary:
    verdict: str = "NOT_RUN"
    group_overlap: int = 0
    top_suspicious: Sequence[str] = ()
    hard_fail_enabled: bool = False

    def to_dict(self) -> LeakageDict:
        return {
            "verdict": self.verdict,
            "group_overlap": int(self.group_overlap),
            "top_suspicious": list(self.top_suspicious),
            "hard_fail": self.hard_fail_enabled and self.verdict.startswith("CRITICAL"),
        }


@dataclass(frozen=True, slots=True)
class ModelQualityReport:
    forecast: ForecastQuality
    decisions: Sequence[DecisionQuality]
    calibration: CalibrationSummary
    advisor_mode: str
    advisor_ablations: Sequence[AdvisorAblationSummary] = ()
    leakage: LeakageSummary = LeakageSummary()

    def to_dict(self) -> ModelQualityReportDict:
        return {
            "forecast": self.forecast.to_dict(),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "calibration": self.calibration.to_dict(),
            "advisor": {
                "mode": self.advisor_mode,
                "ablations": [ablation.to_dict() for ablation in self.advisor_ablations],
            },
            "leakage": self.leakage.to_dict(),
        }


def advisor_ablation_summary(case: AdvisorAblationInput) -> AdvisorAblationSummary:
    guardrail = np.asarray(case.guardrail_decisions)
    proposed = np.asarray(case.proposed_decisions).copy()
    hard_violation = (case.bw_bound > case.bw_hard) | (case.rpc_bound > case.rpc_hard)
    safety_overrides = int(np.sum(hard_violation & (proposed != "HOLD")))
    proposed[hard_violation] = "HOLD"
    changed = int(np.sum(proposed != guardrail))
    return AdvisorAblationSummary(
        mode=case.mode,
        decisions=tuple(str(value) for value in proposed.tolist()),
        changed_from_guardrail=changed,
        safety_overrides=safety_overrides,
    )


def calibration_summary(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> CalibrationSummary:
    covered = (y_true >= lower) & (y_true <= upper)
    widths = np.maximum(upper - lower, 0.0)
    target_names = ("bw", "rpc")
    by_target = {
        target_names[idx] if idx < len(target_names) else f"target_{idx}": float(covered[..., idx].mean())
        for idx in range(y_true.shape[-1])
    }
    by_horizon = {f"h{idx}": float(covered[:, idx, :].mean()) for idx in range(y_true.shape[1])}
    return CalibrationSummary(
        overall_coverage=float(covered.mean()),
        mean_interval_width=float(widths.mean()),
        by_target=by_target,
        by_horizon=by_horizon,
    )
