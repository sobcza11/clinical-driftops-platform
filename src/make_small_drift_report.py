"""
Generates an Evidently Data Drift HTML comparing prepared baseline vs current.
Respects env overrides for input/output paths.
"""
from __future__ import annotations
import os
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

BASELINE = os.getenv("DRIFTOPS_BASELINE_PATH", "data/data_prepared_baseline.csv")
CURRENT = os.getenv("DRIFTOPS_CURRENT_PATH", "data/data_prepared_current.csv")
OUT_HTML = os.getenv("DRIFTOPS_REPORT_PATH", "reports/data_drift_small_demo.html")

def _infer_mapping(df: pd.DataFrame) -> ColumnMapping:
    mapping = ColumnMapping()
    # Treat common time/id columns as categorical/ignored to avoid false drift
    ignore = {"subject_id", "hadm_id", "itemid", "admittime", "charttime", "label"}
    mapping.numerical_features = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    mapping.categorical_features = [c for c in df.columns if c not in ignore and not pd.api.types.is_numeric_dtype(df[c])]
    mapping.target = "label" if "label" in df.columns else None
    return mapping

def build_report(baseline_path: str = BASELINE, current_path: str = CURRENT, out_html: str = OUT_HTML) -> str:
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    df_base = pd.read_csv(baseline_path)
    df_curr = pd.read_csv(current_path)
    mapping = _infer_mapping(df_base)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_base, current_data=df_curr, column_mapping=mapping)
    report.save_html(out_html)
    return out_html

if __name__ == "__main__":
    out = build_report()
    print(f"Evidently drift report saved → {out}")
