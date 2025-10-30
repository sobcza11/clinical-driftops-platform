# src/eval/fairness_audit.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def audit_positive_rate(df: pd.DataFrame, group: str, label: str = "label") -> pd.DataFrame:
    rows = []
    if group not in df.columns or label not in df.columns:
        return pd.DataFrame(columns=["metric", "group", "value"])
    for grp, s in df.groupby(group)[label]:
        try:
            pr = float((pd.to_numeric(s, errors="coerce") == 1).mean())
            rows.append({"metric": "positive_rate", "group": grp, "value": pr})
        except Exception:
            continue
    return pd.DataFrame(rows)

# --- Backward-compatible wrapper ---
def audit_fairness(*args, **kwargs):
    """
    Supports two usages:
      1) audit_fairness(df, group, label="label")
      2) audit_fairness(data_csv, group, out_csv, out_md, label="label")
    """
    # Case 2: paths provided
    if len(args) >= 4 and isinstance(args[0], (str, Path)):
        data_csv, group, out_csv, out_md = args[:4]
        label = kwargs.get("label", "label")
        df = pd.read_csv(data_csv)
        res = audit_positive_rate(df, group, label)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_csv, index=False)
        Path(out_md).write_text(
            "# Fairness Audit (best-effort)\n\n"
            f"- Input: `{data_csv}`\n- Group: `{group}`\n- Label: `{label}`\n\n"
            + ("## Positive Rate by Group\n\n" + res.to_markdown(index=False) if not res.empty
               else "_No valid groups or label column found._"),
            encoding="utf-8"
        )
        return res
    # Case 1: DataFrame provided
    elif len(args) >= 2 and hasattr(args[0], "columns"):
        df, group = args[:2]
        label = kwargs.get("label", "label")
        return audit_positive_rate(df, group, label)
    else:
        raise TypeError("audit_fairness() expects either (df, group, label=...) or (data_csv, group, out_csv, out_md, label=...).")

def main() -> int:
    ap = argparse.ArgumentParser(description="Compute fairness metrics by group.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--label", default="label")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()
    try:
        audit_fairness(args.data, args.group, args.out_csv, args.out_md, label=args.label)
        print(f"✅ fairness metrics → {args.out_csv}")
        print(f"✅ fairness report  → {args.out_md}")
        return 0
    except Exception as e:
        pd.DataFrame(columns=["metric","group","value"]).to_csv(args.out_csv, index=False)
        Path(args.out_md).write_text(
            "# Fairness Audit (best-effort)\n\n"
            f"_Audit failed with error: {e}_\n", encoding="utf-8"
        )
        print(f"⚠️ Fairness audit failed: {e}")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
