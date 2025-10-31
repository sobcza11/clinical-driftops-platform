"""
Lightweight SHAP summary helper (defensive; OK if SHAP is not available).
This module only provides an optional utility and is not required by tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ShapTopFeature:
    name: str
    mean_abs_impact: float


@dataclass
class ShapSummary:
    features: List[Dict[str, Any]]


def compute_top_features(
    shap_values: Optional[Any],
    feature_names: Optional[List[str]],
    topk: int = 10,
) -> ShapSummary:
    """Return an empty summary if inputs are missing; avoids hard deps on shap/numpy."""
    try:
        import numpy as np  # import inside function is intentional
    except Exception:
        return ShapSummary(features=[])

    if shap_values is None or not feature_names:
        return ShapSummary(features=[])

    try:
        # Normalize list-like to array
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values

        sv = np.asarray(sv)
        if sv.ndim == 1:
            sv = sv.reshape(-1, 1)

        mean_abs = np.mean(np.abs(sv), axis=0)  # (n_features,)
        order = np.argsort(mean_abs)[::-1][: max(1, int(topk))]

        feats: List[Dict[str, Any]] = []
        for idx in order:
            i = int(idx)
            if 0 <= i < len(feature_names):
                feats.append(
                    {
                        "name": feature_names[i],
                        "mean_abs_impact": float(mean_abs[i]),
                    }
                )

        return ShapSummary(features=feats)
    except Exception:
        return ShapSummary(features=[])
