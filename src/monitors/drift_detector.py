# -*- coding: utf-8 -*-
# Clinical DriftOps – minimal, safe version

import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy import stats

def _psi_for_arrays(base: np.ndarray, curr: np.ndarray, n_bins: int = 10) -> float:
    base = base[~np.isnan(base)]
    curr = curr[~np.isnan(curr)]
    if base.size == 0 or curr.size == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, n_bins + 1)
    cuts = np.unique(np.quantile(base, quantiles))
    if cuts.size < 2:
        return 0.0

    base_hist, _ = np.histogram(base, bins=cuts)
    curr_hist, _ = np.histogram(curr, bins=cuts)

    base_pct = base_hist / max(base_hist.sum(), 1)
    curr_pct = curr_hist / max(curr_hist.sum(), 1)

    eps = 1e-8
    diff = (curr_pct + eps) - (base_pct + eps)
    ratio = (curr_pct + eps) / (base_pct + eps)
    psi = np.sum(diff * np.log(ratio))
    return float(psi)

def compute_psi(base: pd.Series, curr: pd.Series, n_bins: int = 10) -> float:
    b = pd.to_numeric(base, errors="coerce").to_numpy()
    c = pd.to_numeric(curr, errors="coerce").to_numpy()
    return _psi_for_arrays(b, c, n_bins=n_bins)

def ks_test_feature(base: pd.Series, curr: pd.Series):
    b = pd.to_numeric(base, errors="coerce").dropna()
    c = pd.to_numeric(curr, errors="coerce").dropna()
    if b.empty or c.empty:
        return float("nan"), float("nan")
    d_stat, p_val = stats.ks_2samp(b, c, alternative="two-sided", mode="auto")
    return float(d_stat), float(p_val)

def compare_dataframes(
    df_base: pd.DataFrame,
    df_curr: pd.DataFrame,
    id_cols: Tuple[str, ...] = ("subject_id", "hadm_id", "itemid", "admittime", "charttime"),
) -> pd.DataFrame:
    common_cols = [c for c in df_base.columns.intersection(df_curr.columns) if c not in id_cols]
    numeric_cols = [
        c for c in common_cols
        if pd.api.types.is_numeric_dtype(df_base[c]) or pd.api.types.is_numeric_dtype(df_curr[c])
    ]

    rows: List[dict] = []
    for col in numeric_cols:
        psi = compute_psi(df_base[col], df_curr[col])
        ks_stat, ks_p = ks_test_feature(df_base[col], df_curr[col])
        rows.append(
            {"feature": col, "psi": psi, "ks_stat": ks_stat, "ks_pvalue": ks_p,
             "drift_flag": (psi is not None and not pd.isna(psi) and psi >= 0.2) or
                           (ks_p is not None and not pd.isna(ks_p) and ks_p < 0.01)}
        )
    df = pd.DataFrame(rows).sort_values(["drift_flag", "psi", "ks_stat"], ascending=[False, False, False])
    return df

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="data/data_prepared_baseline.csv")
    p.add_argument("--current", default="data/data_prepared_current.csv")
    args = p.parse_args()

    dfb = pd.read_csv(args.baseline)
    dfc = pd.read_csv(args.current)
    out = compare_dataframes(dfb, dfc)
    flagged = int(out["drift_flag"].sum()) if not out.empty else 0
    print(f"Drift Summary: {flagged}/{len(out)} features flagged.")
    out_path = "reports/drift_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved metrics → {out_path}")

    # ---- MLflow logging (inside CLI, after metrics exist) ----
    try:
        from src.ops.mlflow_tracking import start_run, log_params, log_metrics, log_artifact
        psi_max = float(out["psi"].max(skipna=True)) if not out.empty else 0.0
        ks_max  = float(out["ks_stat"].max(skipna=True)) if not out.empty else 0.0
        with start_run(run_name="drift-detector", tags={"phase": "VI"}):
            log_params({"dataset": "baseline_vs_current", "n_features": len(out)})
            log_metrics({"psi_max": psi_max, "ks_max": ks_max, "drift_flagged": flagged})
            log_artifact(out_path)
            # Attach Evidently HTML if your pipeline produces it:
            html_path = "reports/data_drift_small_demo.html"
            import os
            if os.path.exists(html_path):
                log_artifact(html_path)
    except Exception as e:
        print(f"[mlflow] skipping logging ({e})")

if __name__ == "__main__":
    _cli()

