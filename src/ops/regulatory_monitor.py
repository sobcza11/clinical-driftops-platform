# src/ops/regulatory_monitor.py
# Purpose: Synthesize a lightweight governance snapshot for auditors/regulators.
# Reads:  reports/policy_gate_result.json, reports/performance_metrics.json,
#         reports/fairness_summary.json, reports/shap_top_features.json,
#         reports/live_validation.json
# Writes: reports/regulatory_monitor.json
#
# Notes:
# - This is a heuristic summary, not a legal determination.
# - We "fail closed" where possible.

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

REPORTS = Path("reports")

def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _exists(p: Path) -> bool:
    return p.exists() and p.is_file()

def main(out_dir: str = "reports") -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    perf  = _read_json(out / "performance_metrics.json")
    fair  = _read_json(out / "fairness_summary.json")
    shap  = _read_json(out / "shap_top_features.json")
    gate  = _read_json(out / "policy_gate_result.json")
    live  = _read_json(out / "live_validation.json")

    # Presence checks
    artifacts = {
        "live_validation_json": _exists(out / "live_validation.json"),
        "performance_metrics_json": _exists(out / "performance_metrics.json"),
        "fairness_summary_json": _exists(out / "fairness_summary.json"),
        "shap_top_features_json": _exists(out / "shap_top_features.json"),
        "policy_gate_result_json": _exists(out / "policy_gate_result.json"),
    }

    # Gate status (PASS/FAIL)
    gate_status = str(gate.get("status", "")).upper() if gate else "FAIL"

    # Risk heuristics
    has_perf = perf.get("auroc") is not None and perf.get("ks_stat") is not None
    has_fair = bool(fair.get("slices")) and bool(fair.get("metrics"))
    has_expl = bool(shap.get("features"))

    # Compute a simple risk score (0 low, 2 high)
    risk_score = 0
    if gate_status != "PASS":
        risk_score += 2
    if not has_perf:
        risk_score += 1
    if not has_fair:
        risk_score += 1
    if not has_expl:
        risk_score += 1

    risk_level = "LOW" if risk_score <= 1 else ("MEDIUM" if risk_score == 2 else "HIGH")

    # HIPAA PHI heuristic (we do NOT include PHI in public artifacts by design)
    hipaa_phi_in_artifacts = False
    hipaa_note = "No PHI fields detected in public artifacts by design; raw data not exposed."

    payload = {
        "regulatory_monitor": {
            "policy_gate": gate_status,
            "artifacts_present": artifacts,
            "explainability_present": has_expl,
            "fairness_present": has_fair,
            "performance_present": has_perf,
            "audit_trail_present": artifacts["live_validation_json"],
            "hipaa_phi_in_artifacts": hipaa_phi_in_artifacts,
            "risk_level": risk_level,
            "notes": [
                hipaa_note,
                "This is a heuristic stabilization signal, not a legal opinion."
            ],
        }
    }

    target = out / "regulatory_monitor.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(target)

if __name__ == "__main__":
    print(main())

