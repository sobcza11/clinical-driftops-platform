# monitors/drift_detector.py — REPLACE ENTIRE FILE
from __future__ import annotations
from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual   = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    cuts = np.linspace(np.nanpercentile(expected, 1), np.nanpercentile(expected, 99), bins + 1)
    e_hist, _ = np.histogram(expected, bins=cuts)
    a_hist, _ = np.histogram(actual, bins=cuts)
    e_ratio = np.clip(e_hist / max(e_hist.sum(), 1), 1e-6, 1)
    a_ratio = np.clip(a_hist / max(a_hist.sum(), 1), 1e-6, 1)
    psi = np.sum((a_ratio - e_ratio) * np.log(a_ratio / e_ratio))
    return float(psi) if np.isfinite(psi) else 0.0

def compare_dataframes(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    ignore_cols: Optional[Iterable[str]] = None,
    **kwargs,
) -> Dict[str, float]:
    # Legacy kw support
    if "id_cols" in kwargs and ignore_cols is None:
        ignore_cols = kwargs["id_cols"]
    ign = set([c.lower() for c in (ignore_cols or [])])

    def looks_like_id_or_time(c: str) -> bool:
        cl = c.lower()
        return cl.endswith("_id") or ("time" in cl) or ("date" in cl)

    cols = []
    for c in baseline.columns:
        if c not in current.columns:
            continue
        if looks_like_id_or_time(c) or c.lower() in ign:
            continue
        if pd.api.types.is_numeric_dtype(baseline[c]) and pd.api.types.is_numeric_dtype(current[c]):
            cols.append(c)

    if not cols:
        return {"max_psi": None, "max_ks": None}

    psi_vals, ks_vals = [], []
    for c in cols:
        b = pd.to_numeric(baseline[c], errors="coerce").values
        a = pd.to_numeric(current[c],  errors="coerce").values
        b = b[np.isfinite(b)]; a = a[np.isfinite(a)]
        if len(b) == 0 or len(a) == 0:
            continue
        psi_vals.append(_psi(b, a))
        ks = ks_2samp(b, a).statistic if (len(b) and len(a)) else np.nan
        if np.isfinite(ks):
            ks_vals.append(float(ks))

    max_psi = float(np.nanmax(psi_vals)) if psi_vals else None
    max_ks  = float(np.nanmax(ks_vals))  if ks_vals  else None
    return {"max_psi": max_psi, "max_ks": max_ks}


