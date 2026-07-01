"""Backtest orchestration: load bundle, evaluate on test split, write reports."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from hpc_io_scheduler.config import Config
from hpc_io_scheduler.data.loader import JOB_CAT, JOB_NUM, load_all
from hpc_io_scheduler.data.splits import (
    align_jobs_to_windows,
    cutoff_from_jobs,
)
from hpc_io_scheduler.data.windowing import (
    fit_system_scalers,
    make_windows,
    split_windows_by_horizon,
)
from hpc_io_scheduler.evaluation.audit import check_xgb_leakage
from hpc_io_scheduler.evaluation.baselines import (
    fcfs,
    forecast_point,
    forecast_prob,
    ground_truth_congestion,
)
from hpc_io_scheduler.evaluation.metrics import (
    classifier_metrics,
    decision_metrics,
    pr_auc,
)
from hpc_io_scheduler.evaluation.simulator import simulate
from hpc_io_scheduler.guardrail.kalman import run_causal_kf
from hpc_io_scheduler.guardrail.policy import ForecastGuardrail
from hpc_io_scheduler.guardrail.thresholds import Thresholds
from hpc_io_scheduler.models.congestion import (
    build_rich_features,
    select_best_idx,
)
from hpc_io_scheduler.models.dlinear import ProbabilisticDLinear, predict_dlinear
from hpc_io_scheduler.models.nbeats import NBeatsProbabilistic, predict_nbeats
from hpc_io_scheduler.models.lora_advisor import QwenAdvisor, heuristic_advise
from hpc_io_scheduler.models.xgb_delta import predict_xgb, weighted_rmse


def forecaster_is_dlinear(model) -> bool:
    return isinstance(model, ProbabilisticDLinear)


def _load_bundle(cfg: Config, root: Path) -> tuple:
    md = root / "models"
    sx = joblib.load(md / "scaler_system.joblib")
    sy = joblib.load(md / "scaler_y.joblib")
    pre = joblib.load(md / "preprocessor_job.joblib")
    thr = Thresholds.load(str(root / "guardrail_config.json"))
    kind_path = md / "forecaster_kind.txt"
    kind = kind_path.read_text().strip() if kind_path.exists() else "dlinear"
    if kind == "nbeats":
        from hpc_io_scheduler.models.nbeats import NBeatsProbabilistic
        dlin = NBeatsProbabilistic(
            cfg.data.past_bins, cfg.data.future_bins,
            n_features=len(SYS_FEATURES_FOR_LOAD), n_targets=2, target_idx=(0, 4),
        )
        dlin.load_state_dict(torch.load(md / "nbeats_t1.pt", map_location="cpu"))
    else:
        dlin = ProbabilisticDLinear(
            cfg.data.past_bins, cfg.data.future_bins,
            n_features=len(SYS_FEATURES_FOR_LOAD), n_targets=2, target_idx=(0, 4),
        )
        dlin.load_state_dict(torch.load(md / "dlinear_t1.pt", map_location="cpu"))
    dlin.eval()
    xgb_bw = xgb.Booster(); xgb_bw.load_model(str(md / "xgb_bw.json"))
    xgb_rpc = xgb.Booster(); xgb_rpc.load_model(str(md / "xgb_rpc.json"))
    return sx, sy, pre, thr, dlin, xgb_bw, xgb_rpc


SYS_FEATURES_FOR_LOAD = [
    "bw_recon_mbps", "io_p95", "io_max", "io_fano",
    "lustre_rpc", "active_jobs", "sum_file_rw",
]


def order_to_inv(order: np.ndarray) -> np.ndarray:
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return inv


def job_train_dnn_labels(job_df: pd.DataFrame) -> list[str]:
    train = job_df[job_df["split"] == "train"]
    return sorted(train["dnn_label"].astype(str).unique().tolist())


def _maybe_audit(cfg: Config, job_df: pd.DataFrame, report_dir: Path) -> None:
    """Run XGBoost leakage audit on training fold; raise on CRITICAL."""
    job_train = job_df[job_df["split"] == "train"].reset_index(drop=True)
    pre = joblib.load(Path(cfg.artifact_dir) / "models" / "preprocessor_job.joblib")
    Xj = pre.transform(job_train).astype(np.float32)
    groups = job_train["id_job_norm"].str.split("_").str[0].values
    w = np.where(job_train["dnn_label"] != "unlabeled", 5.0, 1.0)
    feat_names = JOB_NUM + [f"dnn_label_{c}" for c in job_train["dnn_label"].unique()]
    audit = check_xgb_leakage(Xj, job_train["delta_bw_p90"].values, w, groups, cfg, feat_names)
    (report_dir / "xgb_leak_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    verdict = audit.get("verdict", "OK")
    print(f"[audit] {verdict}")
    if verdict.startswith("CRITICAL"):
        raise SystemExit(2)


def run_backtest(cfg: Config, artifact_dir: str = "artifacts",
                 report_dir: str = "reports", run_audit: bool = False) -> dict:
    """Full backtest. Returns dict of metric tables for programmatic use."""
    sys_df, job_df, _ = load_all(cfg)
    cutoff = cutoff_from_jobs(job_df)
    report_path = Path(report_dir); report_path.mkdir(parents=True, exist_ok=True)

    if run_audit:
        _maybe_audit(cfg, job_df, report_path)

    sx, sy, pre, thr, dlin, xgb_bw, xgb_rpc = _load_bundle(cfg, Path(artifact_dir))
    conformal_path = Path(artifact_dir) / "models" / "conformal.joblib"
    conformal = joblib.load(conformal_path) if conformal_path.exists() else None
    guard = ForecastGuardrail(dlin, xgb_bw, xgb_rpc, pre, thr, cfg.guardrail, sy,
                              conformal=conformal)

    # Forecast test
    X, Y, t_pred = make_windows(sys_df, sx, sy, cfg)
    Xtr, Ytr, Xte, Yte, t_te = split_windows_by_horizon(
        X, Y, t_pred, cutoff, horizon_min=5 * cfg.data.future_bins,
    )
    m, s = predict_dlinear(dlin, Xte) if forecaster_is_dlinear(dlin) else predict_nbeats(dlin, Xte)
    shp = m.shape
    m_r = sy.inverse_transform(m.reshape(-1, 2)).reshape(shp)
    s_r = s * sy.scale_
    y_r = sy.inverse_transform(Yte.reshape(-1, 2)).reshape(shp)

    # Causal KF on h=1 (first future bin)
    order = np.argsort(t_te.values)
    inv_t = order_to_inv(order)
    pred_bw = m_r[:, 0, 0][order]; true_bw = y_r[:, 0, 0][order]
    pred_rpc = m_r[:, 0, 1][order]; true_rpc = y_r[:, 0, 1][order]
    bias_bw_ord, P_bw_ord, _, _ = run_causal_kf(
        pred_bw, true_bw,
        sigma_w2=cfg.guardrail.kf_sigma_w2_bw,
        sigma_v2=cfg.guardrail.kf_sigma_v2_bw, P0=cfg.guardrail.kf_p0_bw,
    )
    bias_rpc_ord, P_rpc_ord, _, _ = run_causal_kf(
        pred_rpc, true_rpc,
        sigma_w2=cfg.guardrail.kf_sigma_w2_rpc,
        sigma_v2=cfg.guardrail.kf_sigma_v2_rpc, P0=cfg.guardrail.kf_p0_rpc,
    )
    bias_bw = bias_bw_ord[inv_t]; P_bw = P_bw_ord[inv_t]
    bias_rpc = bias_rpc_ord[inv_t]; P_rpc = P_rpc_ord[inv_t]

    # Align jobs to windows
    job_test = job_df[job_df["split"] == "test"].reset_index(drop=True)
    pos, valid = align_jobs_to_windows(
        job_test["t_start"].values.astype("datetime64[ns]"),
        t_te.values.astype("datetime64[ns]"),
    )
    pos_valid = pos[valid]
    job_test_valid = job_test[valid].reset_index(drop=True)
    Xte_sorted = Xte[order][inv_t][pos_valid]   # restore original test order, then pick pos_valid
    bias_bw_v = bias_bw[pos_valid]; P_bw_v = P_bw[pos_valid]
    bias_rpc_v = bias_rpc[pos_valid]; P_rpc_v = P_rpc[pos_valid]

    prio_arr = pd.to_numeric(
        job_test_valid.get("priority", pd.Series([100] * len(job_test_valid))),
        errors="coerce",
    ).fillna(100).values

    dec, bwb, rpcb, gray = guard.evaluate(
        Xte_sorted, job_test_valid, bias_bw_v, P_bw_v, bias_rpc_v, P_rpc_v,
        priorities=prio_arr,
    )

    # Ground truth (oracle, only for evaluation)
    inv_t = order_to_inv(order)
    s_sorted = s_r[order][inv_t]
    y_sorted = y_r[order][inv_t]
    m_sorted = m_r[order][inv_t]
    bg_bw_max = y_sorted[pos_valid, :, 0].max(1)
    bg_rpc_max = y_sorted[pos_valid, :, 1].max(1)
    job_dbw = job_test_valid["delta_bw_p90"].values
    job_drpc = job_test_valid["delta_rpc_p90"].values
    gt_cong, _ = ground_truth_congestion(
        bg_bw_max, bg_rpc_max, job_dbw, job_drpc,
        thr.bw_hard, thr.rpc_hard, use_hard=True,
    )

    # Predicted deltas for baselines (no oracle)
    Xj_te = pre.transform(job_test_valid).astype(np.float32)
    dmat = xgb.DMatrix(Xj_te)
    delta_bw_pred = xgb_bw.predict(dmat)
    delta_rpc_pred = xgb_rpc.predict(dmat)
    pred_bw_max = m_sorted[pos_valid, :, 0].max(1)
    pred_rpc_max = m_sorted[pos_valid, :, 1].max(1)
    predU_bw_max = (m_sorted[pos_valid, :, 0] + cfg.guardrail.z * s_sorted[pos_valid, :, 0]).max(1)
    predU_rpc_max = (m_sorted[pos_valid, :, 1] + cfg.guardrail.z * s_sorted[pos_valid, :, 1]).max(1)

    n = len(job_test_valid)
    baselines = {
        "FCFS": fcfs(n),
        "Forecast-only (point)": forecast_point(
            pred_bw_max, pred_rpc_max, delta_bw_pred, delta_rpc_pred,
            thr.bw_soft, thr.bw_hard, thr.rpc_soft, thr.rpc_hard,
        ),
        "Forecast-only (prob)": forecast_prob(
            predU_bw_max, predU_rpc_max, delta_bw_pred, delta_rpc_pred,
            thr.bw_soft, thr.bw_hard, thr.rpc_soft, thr.rpc_hard,
        ),
        "FIAC (full)": dec,
    }

    # Gray-zone advisor
    advisor = QwenAdvisor(cfg)
    gray_idx = np.where(gray)[0]
    if len(gray_idx) > 0 and advisor.ok:
        ctx_l = [{
            "job_id": str(job_test_valid.iloc[i]["id_job_norm"]),
            "task_type": str(job_test_valid.iloc[i]["dnn_label"]),
            "cpus_req": int(job_test_valid.iloc[i].get("cpus_req", 0)),
            "sys_status": "GRAY",
            "bw_bound": float(bwb[i]), "rpc_bound": float(rpcb[i]),
            "bw_hard": thr.bw_hard, "rpc_hard": thr.rpc_hard,
            "priority": 100,
        } for i in gray_idx]
        res = advisor.advise_batch(ctx_l, batch_size=cfg.llm.batch_size)
        for j, i in enumerate(gray_idx):
            a, _ = res[j]
            if a is None:
                a, _ = heuristic_advise(ctx_l[j])
            dec[i] = a
        # Safety override
        dec = guard.authority_override(dec, bwb, rpcb, thr.bw_hard, thr.rpc_hard)
        baselines["FIAC (full)"] = dec

    # Decision metrics
    dec_rows = []
    for name, dv in baselines.items():
        r = {"Method": name}
        r.update(decision_metrics(dv, gt_cong))
        sm = simulate(job_test_valid, dv, gt_cong, sys_df, guard, cfg.sim)
        r.update({k: sm[k] for k in [
            "avg_wait_sec", "avg_queue_len", "throughput_jobs_hr",
            "congestion_rate_pct", "utilisation_pct", "completed_jobs",
        ]})
        dec_rows.append(r)
    decision_df = pd.DataFrame(dec_rows)
    decision_df.to_csv(report_path / "decision_metrics.csv", index=False)
    decision_df.to_csv(report_path / "scheduler_metrics.csv", index=False)

    # Forecast metrics
    bg_bw_eval = y_sorted[pos_valid, :, 0].max(1)
    pred_bw_eval = m_sorted[pos_valid, :, 0].max(1)
    bg_rpc_eval = y_sorted[pos_valid, :, 1].max(1)
    pred_rpc_eval = m_sorted[pos_valid, :, 1].max(1)
    fc_row = {
        "rmse_bw": float(np.sqrt(np.mean((bg_bw_eval - pred_bw_eval) ** 2))),
        "rmse_rpc": float(np.sqrt(np.mean((bg_rpc_eval - pred_rpc_eval) ** 2))),
        "n_eval": n,
        "n_gray": int(gray.sum()),
    }
    (report_path / "forecast_metrics.json").write_text(json.dumps(fc_row, indent=2))

    try:
        from hpc_io_scheduler.evaluation.shap_explain import shap_summary
        from hpc_io_scheduler.data.loader import JOB_NUM
        feat = JOB_NUM + [f"dnn_label_{c}" for c in job_train_dnn_labels(job_df)]
        shap_bw = shap_summary(xgb_bw, Xj_te, feat)
        shap_rpc = shap_summary(xgb_rpc, Xj_te, feat)
        pd.DataFrame(shap_bw["ranking"], columns=["feature", "mean_abs_shap"]).to_csv(
            report_path / "shap_xgb_bw.csv", index=False,
        )
        pd.DataFrame(shap_rpc["ranking"], columns=["feature", "mean_abs_shap"]).to_csv(
            report_path / "shap_xgb_rpc.csv", index=False,
        )
        print(f"[shap] wrote shap_xgb_*.csv to {report_path}")
    except Exception as e:
        print(f"[shap] skipped: {e}")

    print(f"[backtest] n={n} gray={int(gray.sum())} report={report_path}")
    print(decision_df.to_string(index=False))
    return {"decision_df": decision_df, "forecast": fc_row, "gt_congestion": gt_cong}
