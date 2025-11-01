# src/ops/policy_validator.py
# Purpose: Validate policy thresholds against reports/performance_metrics.json
# Exports: validate_policy()

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


PERF_PATH = Path("reports/performance_metrics.json")
OUT_PATH = Path("reports/policy_gate_result.json")


@dataclass
class Policy:
    min_auroc: Optional[float] = 0.70
    min_ks: Optional[float] = 0.10

    def as_dict(self) -> Dict[str, Any]:
        return {
            "min_auroc": self.min_auroc,
            "min_ks": self.min_ks,
        }


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ensure_reports_dir() -> None:
    Path("reports").mkdir(parents=True, exist_ok=True)


def validate_policy(
    perf_path: Path | str = PERF_PATH,
    out_path: Path | str = OUT_PATH,
    min_auroc: Optional[float] = 0.70,
    min_ks: Optional[float] = 0.10,
) -> Dict[str, Any]:
    """
    Load metrics, compare to thresholds, write policy_gate_result.json, and return the result dict.
    """
    _ensure_reports_dir()
    perf_p = Path(perf_path)
    out_p = Path(out_path)
    perf = _read_json(perf_p)

    auroc = perf.get("auroc")
    ks = perf.get("ks_stat")

    reasons: List[str] = []
    status = "PASS"

    # Check AUROC
    if auroc is None or min_auroc is None:
        reasons.append("AUROC missing")
        status = "FAIL"
    elif float(auroc) < float(min_auroc):
        reasons.append(f"AUROC {auroc} < min_auroc {min_auroc}")
        status = "FAIL"

    # Check KS
    if ks is None or min_ks is None:
        reasons.append("KS missing")
        status = "FAIL"
    elif float(ks) < float(min_ks):
        reasons.append(f"KS {ks} < min_ks {min_ks}")
        status = "FAIL"

    result = {
        "status": status,
        "policy": Policy(min_auroc=min_auroc, min_ks=min_ks).as_dict(),
        "reasons": reasons,
        "inputs": {"performance_metrics_path": str(perf_p)},
    }

    out_p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    # Defaults align with tests & dashboard
    res = validate_policy()
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
