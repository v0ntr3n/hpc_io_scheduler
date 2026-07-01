"""SHAP-based interpretability for the congestion classifier (Tier-3 L)."""
from __future__ import annotations

import numpy as np


def shap_summary(
    booster, X: np.ndarray, feature_names: list[str], max_samples: int = 500
) -> dict:
    """Return global mean |SHAP| ranking. Cheap, no plots."""
    try:
        import shap
    except ImportError as e:
        raise ImportError("pip install shap") from e
    n = min(len(X), max_samples)
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X[:n])
    mean_abs = np.mean(np.abs(sv), axis=0)
    ranking = sorted(zip(feature_names, mean_abs.tolist()), key=lambda x: -x[1])
    return {"ranking": ranking, "explainer": "tree"}
