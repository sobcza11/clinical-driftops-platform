"""
Clinical DriftOps — Fairness Audit (Phase V)


- Computes per-group classification metrics if `label` present and a protected attribute exists (default: `gender`).
- If model scores/probas aren't available, uses a simple logistic regression to produce predictions on the fly.
- Writes CSV of group metrics and a small Markdown summary.


Usage:
python -m src.eval.fairness_audit \
--data data/data_prepared_current.csv \
--group gender \
--out_csv reports/fairness_metrics.csv \
--out_md reports/fairness_report.md
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROTECTED_CANDIDATES = [
"gender", "race", "ethnicity", "insurance", "marital_status"
]

def _pick_group_column(df: pd.DataFrame, preferred: str | None) -> str | None:
    if preferred and preferred in df.columns:
        return preferred
    for c in PROTECTED_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _prepare_xy(df: pd.DataFrame, label_col: str = "label"):
    if label_col not in df.columns:
        return None, None, []
    ignore = {label_col, "subject_id", "hadm_id", "itemid", "admittime", "charttime"}
    feats = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    if not feats:
        return None, None, []
    X = df[feats].copy()
    y = df[label_col].astype(int)
    return X, y, feats

def audit_fairness(data_path: str, group_col: str | None, out_csv: str, out_md: str) -> tuple[str, str]:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    df = pd.read_csv(data_path)
    group_col = _pick_group_column(df, group_col)
    if group_col is None:
        # Write stubs so CI artifacts exist
        pd.DataFrame({"note": ["No protected attribute found; audit skipped."]}).to_csv(out_csv, index=False)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("# Fairness Audit\n\n_No protected attribute available — skipped._\n")
        print("[INFO] Fairness audit skipped: no group column.")
        return out_csv, out_md

    X, y, feats = _prepare_xy(df)
    if X is None:
        pd.DataFrame({"note": ["No label or numeric features; audit skipped."]}).to_csv(out_csv, index=False)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("# Fairness Audit\n\n_No label or numeric features — skipped._\n")
        print("[INFO] Fairness audit skipped: no label/features.")
        return out_csv, out_md

    # Train quick model to produce predictions
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(Xs, y, df[group_col], test_size=0.3, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Overall metrics
    overall = {
        "group": "__overall__",
        "n": int(len(y_te)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_te, proba)),
        "pos_rate": float(pred.mean()),
    }

    rows = [overall]
    for grp, idx in df.iloc[y_te.index].groupby(g_te).groups.items():
        mask = y_te.index.isin(idx)
        if mask.sum() == 0:
            continue
        rows.append({
            "group": str(grp),
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y_te[mask], pred[mask])),
            "precision": float(precision_score(y_te[mask], pred[mask], zero_division=0)),
            "recall": float(recall_score(y_te[mask], pred[mask], zero_division=0)),
            "f1": float(f1_score(y_te[mask], pred[mask], zero_division=0)),
            "auc": float(roc_auc_score(y_te[mask], proba[mask])),
            "pos_rate": float(pred[mask].mean()),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    # Simple markdown report
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Fairness Audit\n\n")
        f.write(f"**Group column:** `{group_col}`\n\n")
        f.write(out_df.to_markdown(index=False))
        f.write("\n")

    print(f"Saved fairness metrics → {out_csv}")
    print(f"Saved fairness report → {out_md}")
    return out_csv, out_md

    if __name__ == "__main__":
        ap = argparse.ArgumentParser()
        ap.add_argument("--data", default="data/data_prepared_current.csv")
        ap.add_argument("--group", default="gender")
        ap.add_argument("--out_csv", default="reports/fairness_metrics.csv")
        ap.add_argument("--out_md", default="reports/fairness_report.md")
        args = ap.parse_args()
        audit_fairness(args.data, args.group, args.out_csv, args.out_md)

