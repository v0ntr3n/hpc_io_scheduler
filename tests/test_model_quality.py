from __future__ import annotations

import numpy as np
import pytest

from hpc_io_scheduler.evaluation.model_quality import (
    AdvisorAblationInput,
    CalibrationSummary,
    DecisionQuality,
    ForecastQuality,
    LeakageSummary,
    ModelQualityReport,
    advisor_ablation_summary,
    calibration_summary,
)


def test_calibration_summary_reports_coverage_and_width() -> None:
    y_true = np.array([[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]])
    lower = y_true - 0.5
    upper = y_true + 1.5
    upper[0, 0, 0] = 0.0

    summary = calibration_summary(y_true, lower, upper)

    assert summary.overall_coverage == pytest.approx(7 / 8)
    assert summary.mean_interval_width == pytest.approx(2.0 - 0.25)
    assert summary.by_target["bw"] == pytest.approx(3 / 4)
    assert summary.by_target["rpc"] == pytest.approx(1.0)
    assert summary.by_horizon["h0"] == pytest.approx(3 / 4)
    assert summary.by_horizon["h1"] == pytest.approx(1.0)


def test_leakage_summary_keeps_hard_fail_optional() -> None:
    summary = LeakageSummary(
        verdict="CRITICAL: group leakage in fold",
        group_overlap=2,
        top_suspicious=["hist_io_mean"],
        hard_fail_enabled=False,
    )

    payload = summary.to_dict()

    assert payload["verdict"] == "CRITICAL: group leakage in fold"
    assert payload["group_overlap"] == 2
    assert payload["top_suspicious"] == ["hist_io_mean"]
    assert payload["hard_fail"] is False


def test_advisor_ablation_summary_applies_safety_override() -> None:
    case = AdvisorAblationInput(
        mode="heuristic",
        guardrail_decisions=("THROTTLE", "THROTTLE", "HOLD"),
        proposed_decisions=("SUBMIT", "SUBMIT", "SUBMIT"),
        bw_bound=np.array([0.5, 3.0, 0.5]),
        rpc_bound=np.array([0.5, 0.5, 4.0]),
        bw_hard=2.0,
        rpc_hard=2.0,
    )

    summary = advisor_ablation_summary(case)
    payload = summary.to_dict()

    assert payload["mode"] == "heuristic"
    assert payload["decisions"] == ["SUBMIT", "HOLD", "HOLD"]
    assert payload["safety_overrides"] == 2
    assert payload["changed_from_guardrail"] == 2


def test_model_quality_report_serializes_expected_sections() -> None:
    report = ModelQualityReport(
        forecast=ForecastQuality(rmse_bw=0.12, rmse_rpc=3.4, n_eval=20, n_gray=2),
        decisions=[
            DecisionQuality(
                method="FIAC",
                decision_acc=0.9,
                false_submit_rate=0.1,
                false_hold_rate=0.2,
                balanced_acc=0.85,
            )
        ],
        calibration=CalibrationSummary(
            overall_coverage=0.95,
            mean_interval_width=1.25,
            by_target={"bw": 0.94, "rpc": 0.96},
            by_horizon={"h0": 0.9},
        ),
        advisor_mode="heuristic",
        advisor_ablations=[
            advisor_ablation_summary(
                AdvisorAblationInput(
                    mode="guardrail",
                    guardrail_decisions=("SUBMIT",),
                    proposed_decisions=("SUBMIT",),
                    bw_bound=np.array([0.5]),
                    rpc_bound=np.array([0.5]),
                    bw_hard=2.0,
                    rpc_hard=2.0,
                )
            )
        ],
        leakage=LeakageSummary(
            verdict="OK",
            group_overlap=0,
            top_suspicious=[],
            hard_fail_enabled=False,
        ),
    )

    payload = report.to_dict()

    assert payload["forecast"]["rmse_bw"] == pytest.approx(0.12)
    assert payload["decisions"][0]["false_submit_rate"] == pytest.approx(0.1)
    assert payload["calibration"]["by_target"]["rpc"] == pytest.approx(0.96)
    assert payload["calibration"]["by_horizon"]["h0"] == pytest.approx(0.9)
    assert payload["advisor"]["mode"] == "heuristic"
    assert payload["advisor"]["ablations"][0]["mode"] == "guardrail"
    assert payload["leakage"]["verdict"] == "OK"
    assert payload["leakage"]["hard_fail"] is False
