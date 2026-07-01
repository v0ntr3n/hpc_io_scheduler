"""XGBoost delta-IO leakage audit (Tier-2 B).

Original notebook reported RMSE 0.0008 — suspiciously low. This module
checks whether the GroupKFold is correctly using disjoint job groups
and whether `hist_*` features leak future information.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from hpc_io_scheduler.config import Config
from hpc_io_scheduler.data.splits import group_kfold_indices


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def check_xgb_leakage(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
    feature_names: list[str] | None = None,
) -> dict:
    """Two probes:

    1. Group integrity: confirm GroupKFold splits keep `id_job_norm` groups
       disjoint between train/val. If not → leakage, flag CRITICAL.
    2. Per-feature drop: refit with one feature removed; if RMSE jumps
       dramatically, that feature is suspicious (likely leaky).
    """
    audit: dict = {"group_overlap": 0, "rmses": {}}

    # 1. Group integrity
    for fold, (tr, va) in enumerate(group_kfold_indices(groups, cfg.model.xgb_n_splits)):
        gtr = set(groups[tr])
        gva = set(groups[va])
        overlap = gtr & gva
        audit["group_overlap"] = max(audit["group_overlap"], len(overlap))
        if overlap:
            audit[f"fold{fold}_leaked_groups"] = sorted(overlap)[:5]
    if audit["group_overlap"] > 0:
        audit["verdict"] = "CRITICAL: group leakage in fold"

    # 2. Full-fit RMSE
    params = dict(cfg.model.xgb_params, seed=cfg.seed)
    dfull = xgb.DMatrix(X, label=y, weight=w)
    booster = xgb.train(params, dfull, num_boost_round=100)
    full_rmse = _rmse(y, booster.predict(dfull))
    audit["rmses"]["full"] = full_rmse

    # 3. Per-feature drop
    if feature_names is not None:
        drops = {}
        for j, name in enumerate(feature_names):
            mask = np.ones(X.shape[1], dtype=bool)
            mask[j] = False
            params2 = dict(params)
            d2 = xgb.DMatrix(X[:, mask], label=y, weight=w)
            b2 = xgb.train(params2, d2, num_boost_round=100)
            drops[name] = _rmse(y, b2.predict(d2))
        audit["rmses_per_feature"] = drops
        # Top suspicious = biggest *increase* in RMSE
        if drops:
            deltas = {k: v - full_rmse for k, v in drops.items()}
            audit["top_suspicious"] = sorted(deltas, key=lambda k: -deltas[k])[:5]
            audit["verdict"] = audit.get("verdict", "OK")
    return audit
