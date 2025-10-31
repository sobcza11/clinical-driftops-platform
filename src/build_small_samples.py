# src/build_small_samples.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config_mimic_paths import hosp_dir, artifacts_dir

np.random.seed(42)

HOSP = hosp_dir()
OUT = artifacts_dir()
OUT.mkdir(parents=True, exist_ok=True)


def read_csv_any(base: Path, **kwargs):
    """
    Try compressed first, then plain.
    Pass base like HOSP/"patients.csv" (without .gz).
    """
    gz = base.with_suffix(base.suffix + ".gz")  # patients.csv.gz
    if gz.exists():
        return pd.read_csv(gz, **kwargs)
    if base.exists():
        return pd.read_csv(base, **kwargs)
    raise FileNotFoundError(f"Missing: {gz} and {base}")


# ---- 1) Dimension tables (small) ----
patients = read_csv_any(HOSP / "patients.csv")
patients = patients[["subject_id", "gender", "anchor_age", "anchor_year_group"]]

admissions = read_csv_any(
    HOSP / "admissions.csv",
    parse_dates=["admittime", "dischtime"],
    low_memory=False,
)[["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"]]

d_labitems = read_csv_any(HOSP / "d_labitems.csv", low_memory=False)

# ---- 2) Map target labs -> itemids ----
LABELS = ["WHITE BLOOD CELLS", "LACTATE", "CREATININE"]
lab_ids = (
    d_labitems.assign(L=lambda d: d["label"].str.upper())
    .loc[lambda d: d["L"].isin(LABELS), "itemid"]
    .unique()
)
if len(lab_ids) == 0:
    raise RuntimeError(
        "No itemids found for WBC/LACTATE/CREATININE. Check d_labitems labels."
    )

# ---- 3) Stream labevents; filter early; cap rows for demo ----
keep_rows_target = 100_000
cols = ["subject_id", "hadm_id", "charttime", "itemid", "valuenum", "valueuom"]

lab_path_gz = HOSP / "labevents.csv.gz"
lab_path_csv = HOSP / "labevents.csv"
lab_path = lab_path_gz if lab_path_gz.exists() else lab_path_csv
if not lab_path.exists():
    raise FileNotFoundError(f"Missing labevents at {lab_path_gz} or {lab_path_csv}")

filtered = []
kept = 0
for chunk in pd.read_csv(
    lab_path,
    usecols=cols,
    parse_dates=["charttime"],
    chunksize=1_000_000,
    low_memory=False,
):
    c = chunk[chunk["itemid"].isin(lab_ids)].dropna(subset=["valuenum"])
    filtered.append(c)
    kept += len(c)
    if kept >= keep_rows_target:
        break

labs = pd.concat(filtered, ignore_index=True)
# Basic numeric sanity (no-op for valid rows, protects against garbage)
labs = labs[(labs["valuenum"] > -1e6) & (labs["valuenum"] < 1e6)]

# ---- 4) Join to admissions; first 24h window from admittime ----
labs = labs.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
labs = labs[
    (labs["charttime"] >= labs["admittime"])
    & (labs["charttime"] <= labs["admittime"] + pd.Timedelta(hours=24))
]

# ---- 5) Attach demographics (no PHI) ----
labs = labs.merge(patients, on="subject_id", how="left")

# ---- 6) Bring labels and aggregate median per HADM ----
labs = labs.merge(d_labitems[["itemid", "label"]], on="itemid", how="left")
labs["label"] = labs["label"].str.upper()

agg = labs.groupby(
    [
        "subject_id",
        "hadm_id",
        "admission_type",
        "anchor_age",
        "anchor_year_group",
        "gender",
        "label",
    ],
    as_index=False,
).agg(value=("valuenum", "median"))

wide = agg.pivot_table(
    index=[
        "subject_id",
        "hadm_id",
        "admission_type",
        "anchor_age",
        "anchor_year_group",
        "gender",
    ],
    columns="label",
    values="value",
).reset_index()

rename_map = {
    "WHITE BLOOD CELLS": "wbc",
    "LACTATE": "lactate",
    "CREATININE": "creatinine",
}
wide = wide.rename(columns=rename_map)

# Use admittime as time anchor for sorting (one per hadm_id)
hadm_times = admissions[["hadm_id", "admittime"]].drop_duplicates("hadm_id")
wide = wide.merge(hadm_times, on="hadm_id", how="left").rename(
    columns={"admittime": "charttime"}
)

# ---- 7) Canonical schema + sort ----
COLS = [
    "subject_id",
    "hadm_id",
    "anchor_age",
    "gender",
    "admission_type",
    "wbc",
    "lactate",
    "creatinine",
    "charttime",
    "anchor_year_group",
]
wide = wide[[c for c in COLS if c in wide.columns]].copy()
wide["gender"] = wide["gender"].fillna("Unknown")
wide = wide.sort_values(["subject_id", "hadm_id", "charttime"]).reset_index(drop=True)

# ---- 8) Deterministic 50/50 split (demo) ----
n = len(wide)
baseline = wide.iloc[: n // 2].copy()
current = wide.iloc[n // 2 :].copy()

# ---- 9) Save outputs + metadata (CPMAI auditability) ----
baseline_path = OUT / "baseline_sample.csv"
current_path = OUT / "current_sample.csv"
baseline.to_csv(baseline_path, index=False)
current.to_csv(current_path, index=False)

meta = {
    "standard": "PMI-CPMAI Phase II sample build",
    "source_version": "MIMIC-IV v2.2",
    "filters": {
        "labs": ["WBC", "LACTATE", "CREATININE"],
        "window": "first 24h from admittime",
        "rows_scanned_target": keep_rows_target,
    },
    "row_counts": {"baseline": int(len(baseline)), "current": int(len(current))},
    "columns": list(wide.columns),
    "created_at": pd.Timestamp.utcnow().isoformat(),
}
with open(OUT / "baseline_current_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Wrote: {baseline_path} and {current_path}")
print(f"Metadata: {OUT/'baseline_current_meta.json'}")
