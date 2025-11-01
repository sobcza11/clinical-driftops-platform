# explain/shap_summary.py
# Purpose: Normalize SHAP-like feature importance into a standard structure
# Exports: compute_top_features()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_features(raw: Any) -> List[Dict[str, Any]]:
    """
    Accepts several shapes and returns a list of {"name": str, "mean_abs_impact": float}
    Supported inputs:
      - {"features": [{"name": ..., "mean_abs_impact": ...}, ...]}
      - {"top_features": [...same as above...]}
      - list of feature dicts
    Anything else -> []
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
        name = f.get("name")
        imp = f.get("mean_abs_impact")
        if name is None or imp is None:
            # Try alternate keys if present
            name = f.get("feature") if name is None else name
            imp = f.get("importance") if imp is None else imp
        try:
            imp_val = float(imp)
        except Exception:
            continue
        if not isinstance(name, str):
            continue
        out.append({"name": name, "mean_abs_impact": imp_val})

    return out


def compute_top_features(path: str | Path, topk: int = 25) -> List[Dict[str, Any]]:
    """
    Read SHAP summary json-like file and return top-k by mean_abs_impact desc.
    """
    data = _read_json(Path(path))
    feats = _normalize_features(data)
    feats.sort(key=lambda d: d.get("mean_abs_impact", 0.0), reverse=True)
    if topk is not None and topk > 0:
        feats = feats[:topk]
    return feats


__all__ = ["compute_top_features"]
