"""Data loading: parse timestamps, normalize tz, validate schema."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hpc_io_scheduler.config import Config


SYS_FEATURES = [
    "bw_recon_mbps", "io_p95", "io_max", "io_fano",
    "lustre_rpc", "active_jobs", "sum_file_rw",
]
SYS_TARGETS = ["bw_recon_mbps", "lustre_rpc"]

JOB_NUM = [
    "mem_req", "cpus_req", "nodes_alloc", "gres_alloc",
    "hist_io_mean", "hist_startup", "hist_ckpt", "hist_term", "hist_burst",
]
JOB_CAT = ["dnn_label"]


def _parse_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


def load_system(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["bin"] = _parse_ts(df["bin"])
    missing = [c for c in SYS_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"system CSV missing features: {missing}")
    return df.sort_values("bin").reset_index(drop=True)


def load_jobs(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["t_start"] = _parse_ts(df["t_start"])
    df["id_job_norm"] = df["id_job_norm"].astype(str)
    if "split" not in df.columns:
        raise ValueError("job_impact_dataset.csv must have a 'split' column")
    return df


def load_node_data(path: str | Path) -> pd.DataFrame | None:
    """Tier-3 J: node-locality fingerprint. Optional, may be absent."""
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_all(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    base = Path(cfg.data_dir)
    sys_df = load_system(base / cfg.data.sys_csv)
    job_df = load_jobs(base / cfg.data.job_csv)
    node_df = load_node_data(base / cfg.data.node_csv)
    return sys_df, job_df, node_df
