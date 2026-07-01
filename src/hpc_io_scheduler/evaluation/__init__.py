"""Evaluation suite: metrics, baselines, simulator, ablations, audit, SHAP."""
from hpc_io_scheduler.evaluation.ablations import fiac_variant, run_all_ablations
from hpc_io_scheduler.evaluation.audit import check_xgb_leakage
from hpc_io_scheduler.evaluation.baselines import (
    fcfs,
    forecast_point,
    forecast_prob,
    ground_truth_congestion,
)
from hpc_io_scheduler.evaluation.metrics import (
    classifier_metrics,
    coverage,
    decision_metrics,
    forecast_metrics,
    pr_auc,
)
from hpc_io_scheduler.evaluation.model_quality import (
    AdvisorAblationInput,
    AdvisorAblationSummary,
    CalibrationSummary,
    DecisionQuality,
    ForecastQuality,
    LeakageSummary,
    ModelQualityReport,
    advisor_ablation_summary,
    calibration_summary,
)
from hpc_io_scheduler.evaluation.shap_explain import shap_summary
from hpc_io_scheduler.evaluation.simulator import simulate

__all__ = [
    "AdvisorAblationInput",
    "AdvisorAblationSummary",
    "advisor_ablation_summary",
    "check_xgb_leakage",
    "CalibrationSummary",
    "classifier_metrics",
    "coverage",
    "calibration_summary",
    "DecisionQuality",
    "decision_metrics",
    "fiac_variant",
    "fcfs",
    "ForecastQuality",
    "forecast_metrics",
    "forecast_point",
    "forecast_prob",
    "ground_truth_congestion",
    "LeakageSummary",
    "ModelQualityReport",
    "pr_auc",
    "run_all_ablations",
    "shap_summary",
    "simulate",
]
