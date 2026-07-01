"""XGBoost delta-IO estimator (Layer 2). GroupKFold by job_id_prefix.

WARNING: the original notebook showed suspiciously low RMSE (0.0008).
Run `hpc_io_scheduler.evaluation.audit.check_xgb_leakage` after training.
"""
from __future__ import annotations

import time

import numpy as np
import xgboost as xgb

from hpc_io_scheduler.config import Config, XGBParams
from hpc_io_scheduler.data.splits import group_kfold_indices


def weighted_rmse(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    m = w > 1.0
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.average((y[m] - pred[m]) ** 2, weights=w[m])))


def select_best_round(rounds: list[int]) -> int:
    return int(np.median(rounds))


def candidate_xgb_params(cfg: Config) -> list[XGBParams]:
    base = dict(cfg.model.xgb_params, seed=cfg.seed)
    if not cfg.model.xgb_tune or not cfg.model.xgb_tune_grid:
        return [base]
    return [dict(base, **overrides, seed=cfg.seed) for overrides in cfg.model.xgb_tune_grid]


def train_xgb(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
) -> tuple[xgb.Booster, int]:
    """GroupKFold CV; return booster trained on full data with median best_round."""
    candidates = candidate_xgb_params(cfg)
    best_params = candidates[0]
    best_score = float("inf")
    best_rounds: list[int] = []
    for params in candidates:
        rounds: list[int] = []
        scores: list[float] = []
        for tr, va in group_kfold_indices(groups, cfg.model.xgb_n_splits):
            dtr = xgb.DMatrix(X[tr], label=y[tr], weight=w[tr])
            dva = xgb.DMatrix(X[va], label=y[va], weight=w[va])
            bst = xgb.train(
                params,
                dtr,
                num_boost_round=cfg.model.xgb_rounds,
                evals=[(dva, "val")],
                early_stopping_rounds=cfg.model.xgb_early_stop,
                verbose_eval=False,
            )
            pred = bst.predict(dva)
            score = weighted_rmse(y[va], pred, w[va])
            if np.isnan(score):
                score = float(np.sqrt(np.mean((y[va] - pred) ** 2)))
            scores.append(score)
            rounds.append(bst.best_iteration + 1)
        mean_score = float(np.mean(scores))
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
            best_rounds = rounds
    best = select_best_round(best_rounds)
    dfull = xgb.DMatrix(X, label=y, weight=w)
    booster = xgb.train(best_params, dfull, num_boost_round=best)
    return booster, best


def predict_xgb(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    return booster.predict(xgb.DMatrix(X))


def timed_train(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
) -> tuple[xgb.Booster, int, float]:
    t0 = time.time()
    booster, best = train_xgb(X, y, w, groups, cfg)
    return booster, best, time.time() - t0
