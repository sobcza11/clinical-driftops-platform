# src/ops/policy_gate.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

REPORTS = Path("reports")
DEFAULTS = {"min_auroc": 0.70, "min_ks": 0.10}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_thresholds() -> Dict[str, float]:
    return dict(DEFAULTS)


def evaluate(min_auroc: float, min_ks: float, perf: Dict[str, Any]) -> tuple[str, list[str], Optional[float]]:
    auroc = perf.get("auroc")
    ks = perf.get("ks_stat")

    ok = True
    reasons: list[str] = []

    if auroc is None or auroc < min_auroc:
        ok = False
        reasons.append(f"auroc<{min_auroc}")

    if ks is not None and ks < min_ks:
        ok = False
        reasons.append(f"ks<{min_ks}")

    status = "PASS" if ok else "FAIL"
    return status, reasons, ks


def run(reports: Path = REPORTS) -> Dict[str, Any]:
    reports.mkdir(parents=True, exist_ok=True)
    thresholds = _load_thresholds()
    perf = _read_json(reports / "performance_metrics.json") or {}

    status, reasons, ks_val = evaluate(
        min_auroc=thresholds["min_auroc"],
        min_ks=thresholds["min_ks"],
        perf=perf,
    )

    payload: Dict[str, Any] = {
        "status": status,
        "policy": {"min_auroc": thresholds["min_auroc"], "min_ks": thresholds["min_ks"]},
        "reasons": reasons,
        "observed": {"max_psi": None, "max_ks": ks_val},
    }
    (reports / "policy_gate_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def evaluate_policy_gate() -> Dict[str, Any]:
    return run(REPORTS)


def main() -> int:
    run(REPORTS)
    return 0


def entrypoint() -> int:
    return main()





