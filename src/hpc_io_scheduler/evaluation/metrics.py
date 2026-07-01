"""Metrics: forecast, decision, classifier."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    auc,
)


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def classifier_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "threshold": float(thr),
    }


def pr_auc(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    p, r, t = precision_recall_curve(y_true, prob)
    return float(auc(r, p)), p, r, t


def decision_metrics(
    dec: np.ndarray, gt_congestion: np.ndarray
) -> dict:
    """Admit = SUBMIT|THROTTLE. HOLD = inverse."""
    admit = np.isin(dec, ["SUBMIT", "THROTTLE"])
    hold = ~admit
    fsr = float(np.mean(gt_congestion[admit])) if admit.any() else 0.0
    fhr = float(np.mean(~gt_congestion[hold])) if hold.any() else 0.0
    pred_hold = hold.astype(int)
    true_hold = gt_congestion.astype(int)
    acc = float(np.mean(pred_hold == true_hold))
    bal = (
        float(balanced_accuracy_score(true_hold, pred_hold))
        if true_hold.sum() not in (0, len(true_hold))
        else acc
    )
    return {
        "decision_acc": acc,
        "false_submit_rate": fsr,
        "false_hold_rate": fhr,
        "balanced_acc": bal,
    }
