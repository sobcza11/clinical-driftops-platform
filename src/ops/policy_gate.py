# src/ops/policy_gate.py
import sys, json, os
from pathlib import Path
import pandas as pd
import yaml

POLICY = Path("policy.yaml")
DRIFT_CSV = Path("reports/drift_metrics.csv")
FAIR_CSV  = Path("reports/fairness_metrics.csv")
SHAP_PNG  = Path("reports/shap_top_features.png")
RESULT    = Path("reports/policy_gate_result.json")

def main():
    if not POLICY.exists():
        print("policy.yaml not found; failing gate.")
        sys.exit(1)
    policy = yaml.safe_load(POLICY.read_text())

    fails, warns = [], []

    # ---- Drift checks ----
    psi_max = ks_max = 0.0
    if DRIFT_CSV.exists():
        df = pd.read_csv(DRIFT_CSV)
        if "psi" in df:
            psi_max = float(df["psi"].max(skipna=True))
        if "ks_stat" in df:
            ks_max = float(df["ks_stat"].max(skipna=True))

        if psi_max >= policy["drift"]["psi_fail"]:
            fails.append(f"psi_max {psi_max:.3f} >= fail {policy['drift']['psi_fail']}")
        elif psi_max >= policy["drift"]["psi_warn"]:
            warns.append(f"psi_max {psi_max:.3f} >= warn {policy['drift']['psi_warn']}")

        if ks_max >= policy["drift"]["ks_fail"]:
            fails.append(f"ks_max {ks_max:.3f} >= fail {policy['drift']['ks_fail']}")
        elif ks_max >= policy["drift"]["ks_warn"]:
            warns.append(f"ks_max {ks_max:.3f} >= warn {policy['drift']['ks_warn']}")
    else:
        fails.append("missing reports/drift_metrics.csv")

    # ---- Fairness checks ----
    if FAIR_CSV.exists():
        fair = pd.read_csv(FAIR_CSV)
        if "parity_gap" in fair.columns:
            gap = float(fair["parity_gap"].abs().max())
            if gap >= policy["fairness"]["parity_gap_fail"]:
                fails.append(f"parity_gap {gap:.3f} >= fail {policy['fairness']['parity_gap_fail']}")
    else:
        # Fairness optional; no fail if absent, just warn
        warns.append("fairness metrics not found (reports/fairness_metrics.csv)")

    # ---- Explainability checks ----
    if policy.get("explainability", {}).get("require_shap_artifact", True):
        if not SHAP_PNG.exists():
            fails.append("missing explainability artifact reports/shap_top_features.png")

    result = {
        "psi_max": psi_max, "ks_max": ks_max,
        "warns": warns, "fails": fails,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
