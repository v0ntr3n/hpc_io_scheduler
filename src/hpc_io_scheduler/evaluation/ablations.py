"""Ablation runner: drop uncertainty, drop delta-IO, drop reasoning."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
import torch

from hpc_io_scheduler.config import GuardrailConfig
from hpc_io_scheduler.guardrail.kalman import run_causal_kf
from hpc_io_scheduler.evaluation.simulator import simulate
from hpc_io_scheduler.evaluation.metrics import decision_metrics


def fiac_variant(
    *,
    use_uncertainty: bool = True,
    use_delta_io: bool = True,
    use_reasoning: bool = True,
    cache_path: str | None = None,
) -> np.ndarray:
    """Standalone ablation: re-decide without given component.

    Assumes globals: m_real_sorted, s_real_sorted, kf_bw, kf_rpc,
    job_dbw, job_drpc, guardrail, bw_soft/hard, rpc_soft/hard,
    pos_valid, id_arr, label_arr, cpus_arr, bw_mu_arr, rpc_mu_arr,
    prio_arr, advisor.
    """
    from hpc_io_scheduler.models.lora_advisor import heuristic_advise

    z = 1.645 if use_uncertainty else 0.0
    bw = (
        m_real_sorted[pos_valid, :, 0] + kf_bw.r
        + z * np.sqrt(s_real_sorted[pos_valid, :, 0] ** 2 + kf_bw.P)
    ).max(1)
    rp = (
        m_real_sorted[pos_valid, :, 1] + kf_rpc.r
        + z * np.sqrt(s_real_sorted[pos_valid, :, 1] ** 2 + kf_rpc.P)
    ).max(1)
    if use_delta_io:
        bw = bw + job_dbw
        rp = rp + job_drpc
    submit = (bw <= guardrail.bw_soft) & (rp <= guardrail.rpc_soft)
    hold = (bw > guardrail.bw_hard) | (rp > guardrail.rpc_hard)
    out = np.where(submit, "SUBMIT", np.where(hold, "HOLD", "THROTTLE"))

    if not use_reasoning:
        return out

    # Use cached version if present
    if cache_path and pd.io.common.file_exists(cache_path):
        return pd.read_parquet(cache_path)["decision"].values

    gray_idx = np.where(~(submit | hold))[0]
    if len(gray_idx) > 0:
        ctx_l = [
            {
                "job_id": id_arr[i], "task_type": label_arr[i],
                "cpus_req": int(cpus_arr[i]), "sys_status": "GRAY",
                "bw_bound": float(bw[i]), "rpc_bound": float(rp[i]),
                "bw_mu_actual": float(bw_mu_arr[i]),
                "rpc_mu_actual": float(rpc_mu_arr[i]),
                "bw_soft": guardrail.bw_soft, "bw_hard": guardrail.bw_hard,
                "rpc_soft": guardrail.rpc_soft, "rpc_hard": guardrail.rpc_hard,
                "priority": float(prio_arr[i]),
            }
            for i in gray_idx
        ]
        res = advisor.advise_batch(ctx_l, batch_size=32)
        for j, i in enumerate(gray_idx):
            a, _ = res[j]
            if a is None:
                a, _ = heuristic_advise(ctx_l[j])
            out[i] = a

    if cache_path:
        import os
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        pd.DataFrame({"decision": out}).to_parquet(cache_path)
    return out


def run_all_ablations(
    fiac_dec: np.ndarray,
    job_test_valid: pd.DataFrame,
    gt_congestion: np.ndarray,
    sys_state_df: pd.DataFrame,
    guard,
    cfg: SimConfig,
    cache_dir: str,
) -> pd.DataFrame:
    variants = {
        "FIAC (full)": fiac_dec,
        "- uncertainty": fiac_variant(use_uncertainty=False,
                                       cache_path=f"{cache_dir}/abl_uncertainty.parquet"),
        "- DeltaIO": fiac_variant(use_delta_io=False,
                                   cache_path=f"{cache_dir}/abl_dio.parquet"),
        "- reasoning": fiac_variant(use_reasoning=False,
                                     cache_path=f"{cache_dir}/abl_reasoning.parquet"),
    }
    rows = []
    for name, dv in variants.items():
        r = {"Variant": name}
        r.update(decision_metrics(dv, gt_congestion))
        sm = simulate(job_test_valid, dv, gt_congestion, sys_state_df, guard, cfg)
        r["Congestion_Rate"] = sm["congestion_rate_pct"]
        r["Throughput"] = sm["throughput_jobs_hr"]
        rows.append(r)
    return pd.DataFrame(rows)
