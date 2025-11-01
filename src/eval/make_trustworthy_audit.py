# src/eval/make_trustworthy_audit.py
# Purpose: Synthesize reports/trustworthy_audit.json from existing artifacts.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPORTS = Path("reports")


def _read_json(p: Path) -> dict | list:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)

    perf: Dict[str, Any] = _read_json(REPORTS / "performance_metrics.json") or {}
    gate: Dict[str, Any] = _read_json(REPORTS / "policy_gate_result.json") or {}
    shap: Dict[str, Any] = _read_json(REPORTS / "shap_top_features.json") or {}
    history: List[Dict[str, Any]] = _read_json(REPORTS / "drift_history.json") or []

    # summary
    entries = int(perf.get("n") or 0)
    try:
        max_ks = float(perf.get("ks_stat") or 0.0)
    except Exception:
        max_ks = 0.0
    policy_status = (
        (gate.get("status") or "FAIL").upper() if isinstance(gate, dict) else "FAIL"
    )

    # count FAILs in recent drift history (optional)
    drift_flags = 0
    if isinstance(history, list):
        for rec in history[-100:]:
            try:
                if (rec.get("status") or "").upper() == "FAIL":
                    drift_flags += 1
            except Exception:
                pass

    # top shap features
    top_features = []
    if isinstance(shap, dict):
        top_features = shap.get("features") or shap.get("top_features") or []
        if not isinstance(top_features, list):
            top_features = []

    audit = {
        "summary": {
            "entries_evaluated": entries,
            "drift_flags": drift_flags,
            "max_ks": max_ks,
            "policy_status": policy_status,
        },
        "explainability": {
            "top_features": top_features[:25],  # cap for readability
        },
        "sources": [
            "policy_gate_result.json",
            "shap_top_features.json",
            "performance_metrics.json",
            "drift_history.json",
        ],
        "meta": {"psi_alert_threshold": 0.2, "version": "1.0"},
    }

    out = REPORTS / "trustworthy_audit.json"
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    print(main())
