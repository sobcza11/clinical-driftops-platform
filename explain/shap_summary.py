"""
Clinical DriftOps — SHAP Top Feature Summary (Phase V)


- Trains a simple LogisticRegression on prepared data (expects 'label' column; if missing, gracefully exits).
- Computes SHAP values via LinearExplainer (fast, stable for linear models).
- Saves a bar plot of top |mean(|SHAP|)| features to PNG.


Usage:
python -m src.explain.shap_summary \
--data data/data_prepared_current.csv \
--out reports/shap_top_features.png \
--topk 15
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import shap
except Exception as e:
    print("[WARN] SHAP not available:", e)
    shap = None


def _select_features(df: pd.DataFrame, label_col: str = "label"):
    if label_col not in df.columns:
    print(f"[INFO] No '{label_col}' column in data; skipping SHAP.")
    return None, None, None
    ignore = {label_col, "subject_id", "hadm_id", "itemid", "admittime", "charttime"}
    feats = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    if not feats:
    print("[INFO] No numeric features found; skipping SHAP.")
    return None, None, None
    X = df[feats].copy()
    y = df[label_col].astype(int)
    return X, y, feats

def build_and_explain(data_path: str, out_png: str = "reports/shap_top_features.png", topk: int = 15) -> str:
    if shap is None:
        print("[INFO] SHAP not installed; nothing to do.")
        return ""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    df = pd.read_csv(data_path)
    X, y, feats = _select_features(df)
    if X is None:
        # Still write a stub file to keep CI happy
        with open(out_png.replace(".png", "_SKIPPED.txt"), "w", encoding="utf-8") as f:
            f.write("SHAP skipped: missing label or numeric features.\n")
        return ""


    # Train simple baseline model
    scaler = StandardScaler(with_mean=False) # data may already be scaled; keep safe
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)


    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    clf.fit(X_train, y_train)


    # Report AUC for context
    try:
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        print(f"Validation AUC: {auc:.3f}")
    except Exception:
        pass


    # SHAP explainability (linear is efficient)
    try:
        explainer = shap.LinearExplainer(clf, X_train, feature_names=feats)
        shap_values = explainer.shap_values(X_test)
        # For binary, shap_values can be 1D or 2D depending on version
        if isinstance(shap_values, list):
        sv = shap_values[0]
        else:
        sv = shap_values
        mean_abs = np.mean(np.abs(sv), axis=0)
        order = np.argsort(mean_abs)[::-1][:topk]
        top_feats = [feats[i] for i in order]
        top_vals = mean_abs[order]


        # Plot
        plt.figure(figsize=(8, max(3, topk * 0.35)))
        y_pos = np.arange(len(top_feats))
        plt.barh(y_pos, top_vals)
        plt.yticks(y_pos, top_feats)
        plt.gca().invert_yaxis()
        plt.xlabel("mean |SHAP| (impact)")
        plt.title("Top Features by SHAP Impact")
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        print(f"Saved SHAP plot → {out_png}")
        return out_png
    except Exception as e:
        print("[WARN] SHAP computation failed:", e)
        with open(out_png.replace(".png", "_FAILED.txt"), "w", encoding="utf-8") as f:
            f.write(f"SHAP failed: {e}\n")
    return ""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data_prepared_current.csv")
    ap.add_argument("--out", default="reports/shap_top_features.png")
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args()
    build_and_explain(args.data, args.out, args.topk)



