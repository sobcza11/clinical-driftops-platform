# explain/shap_summary.py
# Purpose: Normalize SHAP-like feature importance or compute directly from arrays.
# Exports: compute_top_features() returning an object with ".features" (list[dict])

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    import numpy as np  # optional for array mode
except Exception:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class ShapTop:
    features: List[Dict[str, Any]]


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_features(raw: Any) -> List[Dict[str, Any]]:
    """
    Accepts shapes like:
      - {"features": [{"name": ..., "mean_abs_impact": ...}, ...]}
      - {"top_features": [...same as above...]}
      - list[{"name":..., "mean_abs_impact":...}]
    """
    if isinstance(raw, dict):
        feats = raw.get("features") or raw.get("top_features") or []
    elif isinstance(raw, list):
        feats = raw
    else:
        feats = []

    out: List[Dict[str, Any]] = []
    for f in feats:
        if not isinstance(f, dict):
            continue
        name = f.get("name", f.get("feature"))
        imp = f.get("mean_abs_impact", f.get("importance"))
        if name is None or imp is None:
            continue
        try:
            imp_val = float(imp)
        except Exception:
            continue
        if not isinstance(name, str):
            name = str(name)
        out.append({"name": name, "mean_abs_impact": imp_val})
    return out


def _from_array(
    shap_values: Any,
    feature_names: Optional[Sequence[str]],
    topk: Optional[int],
) -> List[Dict[str, Any]]:
    if np is None:
        return []
    if shap_values is None:
        return []

    arr = np.asarray(shap_values)
    if arr.ndim == 1:
        # single row -> absolute values per feature
        contrib = np.abs(arr)
    else:
        # mean |contribution| over samples
        contrib = np.mean(np.abs(arr), axis=0)

    n = contrib.shape[0]
    names = (
        list(feature_names)
        if feature_names is not None
        else [f"f{i}" for i in range(n)]
    )

    feats = [{"name": n_, "mean_abs_impact": float(v)} for n_, v in zip(names, contrib)]
    feats.sort(key=lambda d: d["mean_abs_impact"], reverse=True)
    if topk is not None and topk > 0:
        feats = feats[:topk]
    return feats


def compute_top_features(
    source: Union[str, Path, List[Dict[str, Any]], None, Any],
    feature_names: Optional[Sequence[str]] = None,
    *,
    topk: int = 25,
) -> ShapTop:
    """
    Flexible entry point expected by tests:

      - From JSON path:
          compute_top_features("reports/shap_top_features.json", topk=25)
      - From dict/list structure:
          compute_top_features({"features":[...]})
      - From raw array + names:
          compute_top_features(np_array, ["f1","f2"], topk=2)
      - Handle None:
          compute_top_features(None, None).features == []
    """
    # Array mode: numpy array or array-like and not a path-like
    if source is not None and not isinstance(source, (str, Path, list, dict)):
        feats = _from_array(source, feature_names, topk)
        return ShapTop(features=feats)

    # None -> empty
    if source is None:
        return ShapTop(features=[])

    # Dict/list structure provided directly
    if isinstance(source, (list, dict)):
        feats = _normalize_features(source)
        feats.sort(key=lambda d: d.get("mean_abs_impact", 0.0), reverse=True)
        if topk is not None and topk > 0:
            feats = feats[:topk]
        return ShapTop(features=feats)

    # Path to JSON
    data = _read_json(Path(source))
    feats = _normalize_features(data)
    feats.sort(key=lambda d: d.get("mean_abs_impact", 0.0), reverse=True)
    if topk is not None and topk > 0:
        feats = feats[:topk]
    return ShapTop(features=feats)


__all__ = ["compute_top_features", "ShapTop"]
