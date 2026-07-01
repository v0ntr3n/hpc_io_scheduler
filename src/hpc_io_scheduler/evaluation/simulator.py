"""Closed-loop event simulator: queue + capacity + waiting-time accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hpc_io_scheduler.config import SimConfig


def simulate(
    jobs_df: pd.DataFrame,
    dec_vec: np.ndarray,
    gt_congestion: np.ndarray,
    sys_state_df: pd.DataFrame,
    guard,
    cfg: SimConfig,
) -> dict:
    """Discrete-time queue with capacity cap; resubmission disabled.

    Returns throughput / wait / queue length / utilisation / congestion.
    """
    jobs = jobs_df.copy()
    jobs["orig_idx"] = np.arange(len(jobs))
    jobs["decision"] = dec_vec
    jobs = jobs.sort_values("t_start").reset_index(drop=True)

    cap = max(int(sys_state_df["active_jobs"].quantile(cfg.capacity_quantile)), cfg.capacity_min)
    rng = np.random.default_rng(cfg.seed)
    durations = np.clip(
        rng.lognormal(np.log(cfg.duration_logn_mean_sec), cfg.duration_logn_sigma, len(jobs)),
        cfg.duration_min_sec,
        cfg.duration_max_sec,
    )

    bins = pd.date_range(jobs["t_start"].min(), jobs["t_start"].max(), freq="5min")
    congested_map = dict(zip(jobs["orig_idx"], gt_congestion))

    pending: list[tuple[pd.Series, int]] = []
    running_end: list[pd.Timestamp] = []
    completed = congested_bins = 0
    total_wait = 0.0
    util_sum = 0.0
    q_len_sum = 0.0

    j = 0
    for now in bins:
        running_end = [e for e in running_end if e > now]
        util_sum += min(len(running_end) / cap, 1.0)
        while j < len(jobs) and jobs.iloc[j]["t_start"] <= now:
            pending.append((jobs.iloc[j], j))
            j += 1
        q_len_sum += len(pending)
        next_q: list[tuple[pd.Series, int]] = []
        for row, orig in pending:
            init_dec = row["decision"]
            if init_dec in ("SUBMIT", "THROTTLE") and len(running_end) < cap:
                end = now + pd.Timedelta(seconds=float(durations[orig]))
                running_end.append(end)
                completed += 1
                total_wait += (now - row["t_start"]).total_seconds()
                if congested_map.get(orig, False):
                    congested_bins += 1
            else:
                next_q.append((row, orig))
        pending = next_q

    span_hr = max((bins[-1] - bins[0]).total_seconds() / 3600.0, 1e-9)
    steps = max(len(bins), 1)
    return {
        "completed_jobs": int(completed),
        "throughput_jobs_hr": float(completed / span_hr),
        "avg_wait_sec": float(total_wait / max(completed, 1)),
        "avg_queue_len": float(q_len_sum / steps),
        "congestion_rate_pct": float(100 * congested_bins / max(completed, 1)),
        "utilisation_pct": float(100 * util_sum / steps),
        "capacity": int(cap),
    }
