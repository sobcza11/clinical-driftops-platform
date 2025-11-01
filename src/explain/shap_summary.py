# src/explain/shap_summary.py
# Purpose: Lightweight SHAP-summary compatibility helper (no heavy deps required).
# It simply normalizes/validates a precomputed JSON of top features (if present).

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ShapSummary:
    features: List[Dict[str, Any]]


def read_top_features(json_path: str | Path) -> ShapSummary:
    """Read a SHAP top-features json and normalize to a list of {name, mean_abs_impact}."""
    p = Path(json_path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ShapSummary(features=[])

    feats = []
    if isinstance(data, dict):
        # Accept either {"features":[...]} or {"top_features":[...]}
        feats = data.get("features") or data.get("top_features") or []
    if not isinstance(feats, list):
        feats = []

    # Normalize entries
    norm: List[Dict[str, Any]] = []
    for item in feats:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        impact = item.get("mean_abs_impact")
        if name is None or impact is None:
            # Try alternate keys
            name = name or item.get("feature") or item.get("column")
            impact = impact or item.get("mean_abs") or item.get("importance")
        try:
            impact = float(impact)
        except Exception:
            continue
        norm.append({"name": str(name), "mean_abs_impact": float(impact)})

    # Stable sort: descending by impact, then by name
    norm.sort(key=lambda d: (-d["mean_abs_impact"], d["name"]))
    return ShapSummary(features=norm)


def write_top_features(
    src_json: str | Path, dst_json: str | Path, topk: int = 25
) -> str:
    """Read, normalize, and write a trimmed top-features file to dst_json."""
    summ = read_top_features(src_json)
    out = Path(dst_json)
    out.write_text(
        json.dumps({"features": summ.features[:topk]}, indent=2), encoding="utf-8"
    )
    return str(out)


if __name__ == "__main__":
    # Simple CLI: python -m src.explain.shap_summary in=reports/shap_top_features.json out=reports/shap_top_features.json
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="reports/shap_top_features.json")
    ap.add_argument("--out", dest="dst", default="reports/shap_top_features.json")
    ap.add_argument("--topk", type=int, default=25)
    args = ap.parse_args()
    print(write_top_features(args.src, args.dst, args.topk))
