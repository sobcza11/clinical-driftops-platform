# monitors/drift_detector.py
from __future__ import annotations
from typing import Dict, Iterable, Optional, List
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    cuts = np.linspace(np.nanpercentile(expected, 1), np.nanpercentile(expected, 99), bins + 1)
    e_hist, _ = np.histogram(expected, bins=cuts)
    a_hist, _ = np.histogram(actual, bins=cuts)
    e_ratio = np.clip(e_hist / max(e_hist.sum(), 1), 1e-6, 1)
    a_ratio = np.clip(a_hist / max(a_hist.sum(), 1), 1e-6, 1)
    psi = np.sum((a_ratio - e_ratio) * np.log(a_ratio / e_ratio))
    return float(psi) if np.isfinite(psi) else 0.0


class DriftSummary:
    def __init__(self, rows: List[Dict[str, float]]):
        self.as_dataframe = pd.DataFrame(rows, columns=["feature", "psi", "ks", "drift_flag"])
        if not self.as_dataframe.empty:
            self.as_dataframe["psi"] = pd.to_numeric(self.as_dataframe["psi"], errors="coerce")
            self.as_dataframe["ks"] = pd.to_numeric(self.as_dataframe["ks"], errors="coerce")
            # Force Python bool in an object-typed column to satisfy `is True`
            self.as_dataframe["drift_flag"] = (
                self.as_dataframe["drift_flag"]
                .map(lambda x: True if bool(x) else False)
                .astype(object)
            )

    @property
    def max_psi(self) -> Optional[float]:
        if self.as_dataframe.empty:
            return None
        v = pd.to_numeric(self.as_dataframe["psi"], errors="coerce").max()
        return float(v) if pd.notna(v) else None

    @property
    def max_ks(self) -> Optional[float]:
        if self.as_dataframe.empty:
            return None
        v = pd.to_numeric(self.as_dataframe["ks"], errors="coerce").max()
        return float(v) if pd.notna(v) else None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {"max_psi": self.max_psi, "max_ks": self.max_ks}


def compare_dataframes(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    ignore_cols: Optional[Iterable[str]] = None,
    **kwargs,
) -> DriftSummary:
    if "id_cols" in kwargs and ignore_cols is None:
        ignore_cols = kwargs["id_cols"]

    ign = set(str(c).lower() for c in (ignore_cols or []))

    def looks_like_id_or_time(c: str) -> bool:
        cl = c.lower()
        return cl.endswith("_id") or "time" in cl or "date" in cl

    cols = []
    for c in baseline.columns:
        if c not in current.columns:
            continue
        if looks_like_id_or_time(c) or c.lower() in ign:
            continue
        if pd.api.types.is_numeric_dtype(baseline[c]) and pd.api.types.is_numeric_dtype(current[c]):
            cols.append(c)

    rows: List[Dict[str, float]] = []
    for c in cols:
        b = pd.to_numeric(baseline[c], errors="coerce").values
        a = pd.to_numeric(current[c], errors="coerce").values
        b = b[np.isfinite(b)]
        a = a[np.isfinite(a)]
        if len(b) == 0 or len(a) == 0:
            continue
        psi = _psi(b, a)
        try:
            ks = float(ks_2samp(b, a).statistic)
        except Exception:
            ks = np.nan

        drift_flag = (psi >= 0.10) or (ks >= 0.10)  # heuristic for tests

        rows.append({
            "feature": c,
            "psi": float(psi) if np.isfinite(psi) else np.nan,
            "ks": float(ks) if np.isfinite(ks) else np.nan,
            "drift_flag": True if drift_flag else False,  # Python bools
        })

    return DriftSummary(rows)

