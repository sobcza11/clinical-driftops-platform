# src/eval/make_trustworthy_audit.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Threshold used when a drift "flag" column is not present.
PSI_ALERT: float = 0.2  # typical "alert" threshold for PSI in many orgs


def _safe_read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file, returning {} if not found or invalid."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV file, returning empty DataFrame if not found or invalid."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _first_existing(reports: Path, candidates: Iterable[str]) -> Optional[Path]:
    """Return the first candidate path that exists under reports/."""
    for name in candidates:
        p = reports / name
        if p.exists():
            return p
    return None


def _load_drift_frame(reports: Path) -> pd.DataFrame:
    """
    Load a drift summary as a DataFrame from common artifacts, trying JSON then CSV.
    We normalize keys to lower-case for easier, case-insensitive access.
    """
    # Try JSON-based artifacts
    json_path = _first_existing(
        reports,
        [
            "drift_history.json",  # list of records
            "small_drift_report.json",  # optional alt
        ],
    )
    if json_path:
        data = _safe_read_json(json_path)
        # Support both list-of-records and dict with "records"/"rows".
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            rows = data.get("records") or data.get("rows") or []
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # If still empty, try CSV variants
    if df.empty:
        csv_path = _first_existing(
            reports,
            [
                "small_drift_report.csv",
                "drift_report.csv",
            ],
        )
        if csv_path:
            df = _safe_read_csv(csv_path)

    # Normalize columns to lower for convenience (keep original df too if needed)
    if not df.empty:
        lower_map = {c: str(c).lower() for c in df.columns}
        df = df.rename(columns=lower_map)

    return df


def _count_drift_flags(df_drift: pd.DataFrame) -> Tuple[int, int]:
    """
    Return (psi_flags, n_rows) where psi_flags is the number of drifted rows/entries.
    If a boolean 'drift_flag' exists, we use its truthiness; otherwise, fall back to PSI>=PSI_ALERT.
    We also look for common column names: 'psi' / 'population_stability_index'.
    """
    if df_drift.empty:
        return 0, 0

    # Resolve candidate columns
    lower = {c.lower(): c for c in df_drift.columns}
    flag_col = lower.get("drift_flag")
    psi_col = lower.get("psi") or lower.get("population_stability_index")

    n_rows = int(df_drift.shape[0])

    if flag_col is not None:
        # Use truthiness (no E712)
        # Coerce to bool where possible; non-bool truthy values counted as True.
        series = df_drift[flag_col]
        try:
            series_bool = series.astype(bool)
        except Exception:
            series_bool = series.apply(bool)
        psi_flags = int(series_bool.sum())
        return psi_flags, n_rows

    if psi_col is not None:
        # Count rows meeting/exceeding PSI threshold.
        try:
            psi_flags = int(
                (
                    pd.to_numeric(df_drift[psi_col], errors="coerce").fillna(0.0)
                    >= PSI_ALERT
                ).sum()
            )
        except Exception:
            psi_flags = 0
        return psi_flags, n_rows

    # If neither column is present, we can’t infer flags.
    return 0, n_rows


def _max_ks(df_drift: pd.DataFrame) -> Optional[float]:
    """
    Return the maximum KS statistic if present. Common column names: 'ks', 'kolmogorov_smirnov'.
    """
    if df_drift.empty:
        return None
    lower = {c.lower(): c for c in df_drift.columns}
    ks_col = lower.get("ks") or lower.get("kolmogorov_smirnov")
    if ks_col is None:
        return None
    try:
        return float(pd.to_numeric(df_drift[ks_col], errors="coerce").max())
    except Exception:
        return None


def _load_policy_gate(reports: Path) -> Dict[str, Any]:
    """Load policy gate result if available."""
    return _safe_read_json(reports / "policy_gate_result.json")


def _load_shap_top(reports: Path, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Load SHAP top features summary. Accepts either:
      {"features": [{"name": "...", "mean_abs_shap": 0.12}, ...]}
    or a bare list. Sorts by the first present of:
      mean_abs_shap, mean_abs_impact, importance (desc).
    """
    raw = _safe_read_json(reports / "shap_top_features.json")
    feats = raw.get("features", raw if isinstance(raw, list) else [])
    if not isinstance(feats, list):
        return []

    def score(rec: Dict[str, Any]) -> float:
        for k in ("mean_abs_shap", "mean_abs_impact", "importance"):
            v = rec.get(k)
            try:
                return float(v)
            except Exception:
                continue
        return 0.0

    try:
        feats = sorted(feats, key=score, reverse=True)
    except Exception:
        pass

    return feats[:limit]


def make_trustworthy_audit(
    reports_dir: str = "reports", out_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a compact 'trustworthy audit' payload from existing artifacts in `reports/`.

    - drift:  uses JSON/CSV candidates to count drift flags and compute max KS
    - policy: reads policy_gate_result.json
    - shap:   reads shap_top_features.json (top features by |SHAP|)

    Writes JSON to `reports/trustworthy_audit.json` (or to `out_path` if provided).
    Returns the payload as a dict.
    """
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    df_drift = _load_drift_frame(reports)
    psi_flags, n_rows = _count_drift_flags(df_drift)
    ks_peak = _max_ks(df_drift)

    policy = _load_policy_gate(reports)
    shap_top = _load_shap_top(reports, limit=10)

    payload: Dict[str, Any] = {
        "summary": {
            "entries_evaluated": n_rows,
            "drift_flags": psi_flags,
            "max_ks": ks_peak,
            "policy_status": policy.get("status"),
        },
        "policy": {
            "status": policy.get("status"),
            "thresholds": policy.get("policy"),
            "reasons": policy.get("reasons", []),
        },
        "explainability": {
            "top_features": shap_top,
        },
        "sources": [
            "policy_gate_result.json",
            "shap_top_features.json",
            # drift sources vary; list the most likely
            "drift_history.json or small_drift_report.(json|csv)",
        ],
        "meta": {
            "psi_alert_threshold": PSI_ALERT,
            "version": "1.0",
        },
    }

    # Determine output file
    out_file = Path(out_path) if out_path else (reports / "trustworthy_audit.json")
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a trustworthy audit JSON from reports/ artifacts."
    )
    parser.add_argument(
        "--reports", default="reports", help="Reports directory (default: reports)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: reports/trustworthy_audit.json)",
    )
    args = parser.parse_args()

    result = make_trustworthy_audit(reports_dir=args.reports, out_path=args.out)
    print(json.dumps(result, indent=2))
