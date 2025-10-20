#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clinical DriftOps Platform — Phase III Data Preparation
"""

from __future__ import annotations
from datetime import datetime, timezone
import argparse
import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---------- Defaults / Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
OUTPUTS = {
    "baseline_in": DATA_DIR / "baseline_sample.csv",
    "current_in": DATA_DIR / "current_sample.csv",
    "baseline_out": DATA_DIR / "data_prepared_baseline.csv",
    "current_out": DATA_DIR / "data_prepared_current.csv",
    "scaler_params": REPORTS_DIR / "data_prep_scaler_params.json",
    "manifest": REPORTS_DIR / "data_prep_meta.json",
}

ID_TIME_COLS = {"subject_id", "hadm_id", "admittime", "charttime", "itemid", "label"}  # preserved, not scaled

# ---------- Helpers ----------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_csv(path: Path) -> pd.DataFrame:
    # Parse timestamps if present in header
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        header = fh.readline()
    parse_cols = [c for c in ["admittime", "charttime"] if c in header]
    return pd.read_csv(path, parse_dates=parse_cols, low_memory=False)

def split_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = list(df.columns)
    id_time = [c for c in cols if c in ID_TIME_COLS]
    num = [c for c in cols if c not in ID_TIME_COLS and pd.api.types.is_numeric_dtype(df[c])]
    return id_time, num

def clip_outliers(df: pd.DataFrame, cols: List[str], method: str, z_thr: float, iqr_mult: float) -> pd.DataFrame:
    if method == "none" or not cols:
        return df
    df = df.copy()
    if method == "zscore":
        for c in cols:
            s = df[c]
            mu, sd = s.mean(), s.std(ddof=0)
            if sd == 0 or np.isnan(sd):
                continue
            lo, hi = mu - z_thr * sd, mu + z_thr * sd
            df[c] = s.clip(lower=lo, upper=hi)
    elif method == "iqr":
        for c in cols:
            s = df[c]
            q1, q3 = np.nanpercentile(s, 25), np.nanpercentile(s, 75)
            iqr = q3 - q1
            lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
            df[c] = s.clip(lower=lo, upper=hi)
    else:
        raise ValueError(f"Unknown outlier method: {method}")
    return df

def fit_scaler(scaler_name: str, X: np.ndarray):
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scaler: {scaler_name}")
    scaler.fit(X)
    return scaler

def apply_scaler(scaler, X: np.ndarray):
    return X if scaler is None else scaler.transform(X)

def df_stats(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for c in cols:
        s = df[c]
        stats[c] = {
            "count": int(s.count()),
            "mean": float(np.nanmean(s)) if len(s) else np.nan,
            "std": float(np.nanstd(s, ddof=0)) if len(s) else np.nan,
            "min": float(np.nanmin(s)) if len(s) else np.nan,
            "p50": float(np.nanpercentile(s, 50)) if len(s) else np.nan,
            "max": float(np.nanmax(s)) if len(s) else np.nan,
            "missing": int(s.isna().sum()),
        }
    return stats

@dataclass
class PrepConfig:
    scaler: str = "standard"         # standard | minmax | none
    outliers: str = "zscore"         # zscore | iqr | none
    z_threshold: float = 4.0         # for zscore
    iqr_mult: float = 1.5            # for iqr
    impute: str = "drop"             # drop | zero | median
    random_seed: int = 42

def impute_frame(df: pd.DataFrame, cols: List[str], strategy: str) -> pd.DataFrame:
    df = df.copy()
    if strategy == "drop":
        return df.dropna(subset=cols)
    elif strategy == "zero":
        for c in cols:
            df[c] = df[c].fillna(0.0)
        return df
    elif strategy == "median":
        med = df[cols].median(numeric_only=True)
        df[cols] = df[cols].fillna(med)
        return df
    else:
        raise ValueError(f"Unknown impute strategy: {strategy}")

# ---------- Main pipeline ----------
def prepare(split_name: str, df: pd.DataFrame, cfg: PrepConfig, scaler=None):
    id_time_cols, num_cols = split_cols(df)
    # 1) Impute
    df1 = impute_frame(df, num_cols, cfg.impute)
    # 2) Outliers
    df2 = clip_outliers(df1, num_cols, cfg.outliers, cfg.z_threshold, cfg.iqr_mult)
    # 3) Scale
    X = df2[num_cols].to_numpy(dtype=float, copy=True)
    if scaler is None:
        scaler = fit_scaler(cfg.scaler, X)
    X_scaled = apply_scaler(scaler, X)
    df2[num_cols] = X_scaled
    stats = {
        "id_time_cols": id_time_cols,
        "num_cols": num_cols,
        "pre_stats": df_stats(df, num_cols),
        "post_stats": df_stats(df2, num_cols),
    }
    return df2, scaler, stats

def main():
    parser = argparse.ArgumentParser(description="Phase III Data Preparation")
    parser.add_argument("--scaler", default="standard", choices=["standard", "minmax", "none"])
    parser.add_argument("--outliers", default="zscore", choices=["zscore", "iqr", "none"])
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--iqr-mult", type=float, default=1.5)
    parser.add_argument("--impute", default="drop", choices=["drop", "zero", "median"])
    args = parser.parse_args()

    cfg = PrepConfig(
        scaler=args.scaler,
        outliers=args.outliers,
        z_threshold=args.z_threshold,
        iqr_mult=args.iqr_mult,
        impute=args.impute,
    )

    np.random.seed(cfg.random_seed)
    ensure_dirs()

    # Load inputs
    for key in ("baseline_in", "current_in"):
        if not OUTPUTS[key].exists():
            raise FileNotFoundError(f"Missing input file: {OUTPUTS[key]}")

    df_base = load_csv(OUTPUTS["baseline_in"])
    df_curr = load_csv(OUTPUTS["current_in"])

    # Prepare baseline (fit), current (apply)
    base_prep, scaler, base_stats = prepare("baseline", df_base, cfg, scaler=None)
    curr_prep, _, curr_stats = prepare("current", df_curr, cfg, scaler=scaler)

    # Reorder columns: id/time first
    def reorder_cols(df):
        id_cols = [c for c in ID_TIME_COLS if c in df.columns]
        other = [c for c in df.columns if c not in id_cols]
        return df[id_cols + other]

    base_prep = reorder_cols(base_prep)
    curr_prep = reorder_cols(curr_prep)

    # Save outputs
    base_prep.to_csv(OUTPUTS["baseline_out"], index=False)
    curr_prep.to_csv(OUTPUTS["current_out"], index=False)

    # Save scaler params (if any)
    scaler_dict = None
    if scaler is not None:
        if isinstance(scaler, StandardScaler):
            scaler_dict = {
                "type": "StandardScaler",
                "mean_": scaler.mean_.tolist(),
                "scale_": scaler.scale_.tolist(),
                "var_": scaler.var_.tolist(),
                "n_features_in_": int(scaler.n_features_in_),
                "feature_names_in_": base_stats["num_cols"],
            }
        elif isinstance(scaler, MinMaxScaler):
            scaler_dict = {
                "type": "MinMaxScaler",
                "data_min_": scaler.data_min_.tolist(),
                "data_max_": scaler.data_max_.tolist(),
                "data_range_": scaler.data_range_.tolist(),
                "n_features_in_": int(scaler.n_features_in_),
                "feature_names_in_": base_stats["num_cols"],
            }

    with open(OUTPUTS["scaler_params"], "w") as f:
        json.dump(scaler_dict, f, indent=2)

    # Manifest
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "inputs": {
            "baseline_sample.csv": str(OUTPUTS["baseline_in"].resolve()),
            "current_sample.csv": str(OUTPUTS["current_in"].resolve()),
        },
        "outputs": {
            "data_prepared_baseline.csv": str(OUTPUTS["baseline_out"].resolve()),
            "data_prepared_current.csv": str(OUTPUTS["current_out"].resolve()),
            "data_prep_scaler_params.json": str(OUTPUTS["scaler_params"].resolve()),
        },
        "hashes": {},
        "stats": {
            "baseline": base_stats,
            "current": curr_stats,
        },
        "notes": [
            "ID/timestamp columns preserved; scaling applied only to numeric features.",
            "Outlier clipping applied before scaling.",
        ],
        "provenance": {
            "script": "src/data_prep.py",
            "version_hint": "v1.0",
        },
    }

    for k in ("baseline_out", "current_out", "scaler_params"):
        p = OUTPUTS[k]
        if p.exists():
            manifest["hashes"][p.name] = sha256(p)

    with open(OUTPUTS["manifest"], "w") as f:
        json.dump(manifest, f, indent=2)

    print("[DataPrep] Complete.")
    print(f"  -> {OUTPUTS['baseline_out'].relative_to(REPO_ROOT)}")
    print(f"  -> {OUTPUTS['current_out'].relative_to(REPO_ROOT)}")
    print(f"  -> {OUTPUTS['scaler_params'].relative_to(REPO_ROOT)}")
    print(f"  -> {OUTPUTS['manifest'].relative_to(REPO_ROOT)}")

if __name__ == "__main__":
    main()