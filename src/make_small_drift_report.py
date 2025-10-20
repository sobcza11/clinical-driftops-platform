#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates a compact Evidently data-drift report comparing prepared baseline vs current.
Outputs: reports/data_drift_small_demo.html
"""
from pathlib import Path
import pandas as pd

# if Evidently not installed: pip install evidently==0.4.36
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
REPORTS = REPO / "reports"

BASE = DATA / "data_prepared_baseline.csv"
CURR = DATA / "data_prepared_current.csv"
OUT = REPORTS / "data_drift_small_demo.html"

def main():
    if not BASE.exists() or not CURR.exists():
        raise FileNotFoundError("Run data_prep.py first to create prepared baseline/current files.")
    REPORTS.mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(BASE, low_memory=False)
    cur = pd.read_csv(CURR, low_memory=False)

    # Drop non-numeric/identifier columns from drift metrics
    id_like = {"subject_id","hadm_id","itemid","label","admittime","charttime"}
    numeric_cols = [c for c in ref.columns if c not in id_like and pd.api.types.is_numeric_dtype(ref[c])]

    ref = ref[numeric_cols].copy()
    cur = cur[numeric_cols].copy()

    r = Report(metrics=[DataDriftPreset()])
    r.run(reference_data=ref, current_data=cur)
    r.save_html(str(OUT))
    print(f"[Drift] Report saved -> {OUT.relative_to(REPO)}")

if __name__ == "__main__":
    main()