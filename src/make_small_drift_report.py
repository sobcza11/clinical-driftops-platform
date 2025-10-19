from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

DATA = Path("data")
REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

baseline = pd.read_csv(DATA / "baseline_sample.csv")
current  = pd.read_csv(DATA / "current_sample.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=baseline, current_data=current)
out = REPORTS / "data_drift_small_demo.html"
report.save_html(out)
print(f"Wrote {out}")
