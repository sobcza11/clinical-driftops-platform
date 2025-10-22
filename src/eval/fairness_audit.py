"""
Run a simple fairness audit by comparing positive rates between groups.
Usage:
    python -m src.eval.fairness_audit --data data/data_prepared_current.csv --group gender --out_csv reports/fairness_metrics.csv --out_md reports/fairness_report.md
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "label" not in df.columns:
        print("⚠️ No 'label' column; audit skipped.")
        return

    if args.group not in df.columns:
        print(f"⚠️ Group column '{args.group}' not found.")
        return

    metrics = (
        df.groupby(args.group)["label"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "positive_rate"})
        .reset_index()
    )

    metrics.to_csv(args.out_csv, index=False)

    # write simple markdown table
    with open(args.out_md, "w") as f:
        f.write(metrics.to_markdown(index=False))

    print(f"✅ fairness metrics → {args.out_csv}")
    print(f"✅ fairness report  → {args.out_md}")

if __name__ == "__main__":
    main()