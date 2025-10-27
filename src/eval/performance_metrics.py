# -*- coding: utf-8 -*-
"""
Compute classification performance metrics from a predictions CSV and emit JSON/CSV.
Expected columns: y_true, y_score  (optionally y_pred; otherwise we derive thresholds)
Outputs:
  reports/performance_metrics.json
  reports/performance_metrics.csv
"""

from __future__ import annotations
import json
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    precision_recall_fscore_support,
    accuracy_score,
)

POSSIBLE_INPUTS = [
    Path("reports/predictions.csv"),
    Path("reports/predictions_current.csv"),
    Path("data/predictions.csv"),
]

OUT_DIR = Path("reports")
JSON_OUT = OUT_DIR / "performance_metrics.json"
CSV_OUT  = OUT_DIR / "performance_metrics.csv"

def _find_input() -> Path:
    for p in POSSIBLE_INPUTS:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No predictions file found. Looked for: {', '.join(map(str, POSSIBLE_INPUTS))}"
    )

def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _clip_probs(arr: np.ndarray) -> np.ndarray:
    # avoid log-loss blowing up with exactly 0/1
    eps = 1e-15
    return np.clip(arr, eps, 1 - eps)

def _threshold_metrics(y_true, y_score, thr: float):
    y_pred = (y_score >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

def _best_f1_threshold(y_true, y_score):
    # scan thresholds at unique scores + 0.5 to be safe on tiny samples
    uniq = np.unique(y_score)
    cands = np.unique(np.concatenate([uniq, np.array([0.5])]))
    best = (-1.0, 0.5, {})  # f1, thr, metrics
    for thr in cands:
        m = _threshold_metrics(y_true, y_score, float(thr))
        if m["f1"] > best[0]:
            best = (m["f1"], float(thr), m)
    return best[1], best[2]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = _find_input()

    df = pd.read_csv(pred_path)
    cols = [c.lower() for c in df.columns]
    rename_map = {old: old.lower() for old in df.columns}
    df = df.rename(columns=rename_map)

    if "y_true" not in df.columns or "y_score" not in df.columns:
        raise ValueError(
            f"Input {pred_path} must contain columns y_true, y_score (found: {list(df.columns)})"
        )

    y_true = _safe_float_series(df["y_true"]).astype(int).to_numpy()
    y_score = _safe_float_series(df["y_score"]).astype(float).to_numpy()

    # Basic sanity
    if len(y_true) == 0 or len(y_score) == 0 or len(y_true) != len(y_score):
        raise ValueError("Invalid predictions: empty or length mismatch for y_true/y_score.")

    # Core metrics
    metrics = {}
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["auroc"] = float("nan")

    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        metrics["auprc"] = float("nan")

    # Log loss (only when probabilities look valid)
    probs = _clip_probs(y_score.copy())
    try:
        metrics["log_loss"] = float(log_loss(y_true, probs))
    except Exception:
        metrics["log_loss"] = float("nan")

    # Thresholded metrics
    # 0.5
    t05 = _threshold_metrics(y_true, y_score, 0.5)
    # best F1
    best_thr, best_m = _best_f1_threshold(y_true, y_score)

    # Flatten for CSV/JSON
    flat = {
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "log_loss": metrics["log_loss"],
        "acc@0.5": t05["accuracy"],
        "prec@0.5": t05["precision"],
        "recall@0.5": t05["recall"],
        "f1@0.5": t05["f1"],
        "best_f1_threshold": best_thr,
        "acc@bestF1": best_m["accuracy"],
        "prec@bestF1": best_m["precision"],
        "recall@bestF1": best_m["recall"],
        "f1@bestF1": best_m["f1"],
        "source_file": str(pred_path),
    }

    # Write JSON
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(flat, f, indent=2)

    # Write CSV (one-row CSV to be easy to read & plot later)
    pd.DataFrame([flat]).to_csv(CSV_OUT, index=False)

    # Console summary for Actions log
    print("== Performance Metrics ==")
    for k in ["n_samples", "positive_rate", "auroc", "auprc", "log_loss", "f1@0.5", "f1@bestF1", "best_f1_threshold"]:
        print(f"{k}: {flat[k]}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[performance_metrics] ERROR: {e}", file=sys.stderr)
        sys.exit(1)