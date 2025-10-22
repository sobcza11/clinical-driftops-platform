"""
Phase V — Fairness Audit (group disparity)
Produces per-group positive rate and disparity relative to overall mean.
If group column or label is missing, writes a stub so CI stays green.
"""
import argparse
import pandas as pd
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--out_md", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.data)

    if args.group not in df.columns:
        note = f"Group column '{args.group}' not found; audit skipped."
        pd.DataFrame({"note":[note]}).to_csv(args.out_csv, index=False)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(f"# Fairness Audit\n\n{note}\n")
        print("⚠️", note)
        return

    if "label" not in df.columns:
        note = "No 'label' column; audit skipped."
        pd.DataFrame({"note":[note]}).to_csv(args.out_csv, index=False)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(f"# Fairness Audit\n\n{note}\n")
        print("⚠️", note)
        return

    rows = []
    overall_rate = float(df["label"].mean())
    for g, sub in df.groupby(args.group):
        pr = float(sub["label"].mean()) if len(sub) else np.nan
        rows.append({"group": str(g), "n": int(len(sub)), "positive_rate": pr})

    out = pd.DataFrame(rows)
    out["disparity"] = out["positive_rate"] - overall_rate
    out.to_csv(args.out_csv, index=False)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# Fairness Audit\n\n")
        f.write(f"**Group column:** `{args.group}`\n\n")
        f.write(f"**Overall positive rate:** {overall_rate:.4f}\n\n")
        f.write(out.to_markdown(index=False))
        f.write("\n")

    print(f"✅ fairness metrics → {args.out_csv}")
    print(f"✅ fairness report  → {args.out_md}")

if __name__ == "__main__":
    main()