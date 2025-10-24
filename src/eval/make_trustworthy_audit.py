"""
Make a consolidated Trustworthy AI Audit (Phase V) as Markdown.

Inputs (expected to exist, but handled gracefully if missing):
- reports/drift_metrics.csv
- reports/fairness_metrics.csv
- reports/fairness_report.md
- reports/data_drift_small_demo.html
- reports/shap_top_features.png

Usage:
  python -m src.eval.make_trustworthy_audit --out reports/trustworthy_ai_audit_v1.md
"""

from __future__ import annotations
import argparse
import os
import pandas as pd
from datetime import datetime

def read_csv_safe(path: str):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ Could not read {path}: {e}")
    return None

def read_text_safe(path: str) -> str | None:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print(f"⚠️ Could not read {path}: {e}")
    return None

def to_md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        # Basic fallback
        header = " | ".join(map(str, df.columns))
        sep = " | ".join(["---"] * len(df.columns))
        rows = [" | ".join(map(str, r)) for r in df.values]
        return "\n".join([header, sep, *rows])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/trustworthy_ai_audit_v1.md")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Inputs
    drift_csv = "reports/drift_metrics.csv"
    fair_csv = "reports/fairness_metrics.csv"
    fair_md  = "reports/fairness_report.md"
    drift_html = "reports/data_drift_small_demo.html"
    shap_png = "reports/shap_top_features.png"

    df_drift = read_csv_safe(drift_csv)
    df_fair  = read_csv_safe(fair_csv)
    md_fair  = read_text_safe(fair_md)

    # Heuristics / thresholds (can later drive CI pass/fail)
    PSI_ALERT = 0.2  # example threshold
    FAIR_GAP  = 0.10

    # Summaries
    psi_flags = 0
    if df_drift is not None:
        # Try common column names; adjust to your actual CSV schema
        # Expecting columns like: feature, psi, ks, p_value, drift_flag
        cols = [c.lower() for c in df_drift.columns]
        lower = {c.lower(): c for c in df_drift.columns}
        psi_col = lower.get("psi") or lower.get("population_stability_index", None)
        flag_col = lower.get("drift_flag") or lower.get("drifted", None)

        if psi_col is not None and flag_col is not None:
            psi_flags = int(df_drift[df_drift[flag_col] == True].shape[0])
        elif psi_col is not None:
            psi_flags = int((df_drift[psi_col] >= PSI_ALERT).sum())

    fair_gaps = None
    if df_fair is not None:
        # Expect columns: group, positive_rate, disparity (may vary)
        # If disparity missing, compute relative to mean positive rate
        lower = {c.lower(): c for c in df_fair.columns}
        if "disparity" in lower:
            fair_gaps = df_fair[lower["disparity"]].abs().max()
        elif "positive_rate" in lower:
            mean_rate = float(df_fair[lower["positive_rate"]].mean())
            fair_gaps = float((df_fair[lower["positive_rate"]] - mean_rate).abs().max())

    # Compose Markdown
    lines = []
    lines.append(f"# Trustworthy AI Audit v1.0 (Phase V)\n")
    lines.append(f"_Generated: {now}_\n")
    lines.append("## Overview")
    lines.append("- **Project:** Clinical DriftOps Platform")
    lines.append("- **Scope:** Phase V – Model Evaluation (drift, explainability, fairness)")
    lines.append("")

    # Drift
    lines.append("## 1) Data Drift Summary")
    if df_drift is None:
        lines.append(f"- Drift metrics CSV not found at `{drift_csv}`.")
    else:
        lines.append(f"- Source: `{drift_csv}`")
        lines.append(f"- Features flagged for drift (heuristic): **{psi_flags}** (PSI ≥ {PSI_ALERT})")
        lines.append("")
        lines.append(to_md_table(df_drift.head(25)))
        lines.append("")
        if os.path.exists(drift_html):
            lines.append(f"- Full Evidently report: `{drift_html}`")
    lines.append("")

    # Explainability
    lines.append("## 2) Explainability (SHAP)")
    if os.path.exists(shap_png):
        lines.append(f"![SHAP Top Features]({shap_png})")
        lines.append("")
    else:
        lines.append(f"- SHAP plot not found at `{shap_png}`.\n")

    # Fairness
    lines.append("## 3) Fairness Audit")
    if df_fair is None and md_fair is None:
        lines.append(f"- Fairness outputs not found at `{fair_csv}` / `{fair_md}`.")
    else:
        if fair_gaps is not None:
            lines.append(f"- Max absolute disparity (heuristic): **{fair_gaps:.3f}** (threshold {FAIR_GAP})")
        if df_fair is not None:
            lines.append("")
            lines.append(to_md_table(df_fair))
            lines.append("")
        if md_fair:
            lines.append("### Fairness Report (Markdown)")
            lines.append("")
            lines.append(md_fair)
    lines.append("")

    # Governance
    lines.append("## 4) Governance Notes")
    lines.append("- HIPAA-aligned (de-identified data), FDA GMLP lineage & reproducibility, EU AI Act monitoring.")
    lines.append("- Human-in-the-loop review recommended for flagged drift or fairness gaps.")
    lines.append("")

    # Save
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Wrote audit → {args.out}")

if __name__ == "__main__":
    main()
