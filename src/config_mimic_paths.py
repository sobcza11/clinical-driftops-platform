# src/config_mimic_paths.py
from pathlib import Path
import os

def mimic_root() -> Path:
    """
    Returns the root folder containing your local MIMIC-IV v2.2 files.
    Priority:
      1) ENV VAR: MIMIC_V2_2_ROOT
      2) Default known path (edit if needed)
    """
    env = os.environ.get("MIMIC_V2_2_ROOT")
    if env:
        return Path(env)

    # EDIT this default if your path differs
    return Path(r"C:\Users\Rand Sobczak Jr\_rts\3_AI\data_mimiciv\v2_2")

def hosp_dir() -> Path:
    return mimic_root() / "hosp"

def icu_dir() -> Path:
    return mimic_root() / "icu"

def artifacts_dir() -> Path:
    # drift-ready CSVs + reports live in repo-local data/reports (gitignored)
    return Path("data")

def reports_dir() -> Path:
    return Path("reports")
