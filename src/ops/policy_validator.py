# src/ops/policy_validator.py
# Purpose: Validate policy thresholds against performance metrics.
# Exports: validate_policy()

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PERF_PATH = Path("reports/performance_metrics.json")
OUT_PATH = Path("reports/policy_gate_result.json")


@dataclass
class Policy:
    min_auroc: Optional[float] = 0.70
    min_ks: Optional[float] = 0.10

    @classmethod
    def from_dict(cls, d: Dict[str, Any] | None) -> "Policy":
        d = d or {}
        return cls(
            min_auroc=d.get("min_auroc", 0.70),
            min_ks=d.get("min_ks", 0.10),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {"min_auroc": self.min_auroc, "min_ks": self.min_ks}


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ensure_reports_dir() -> None:
    Path("reports").mkdir(parents=True, exist_ok=True)


def _as_perf_dict(
    perf_or_path: Union[Dict[str, Any], str, Path, None],
) -> Dict[str, Any]:
    if isinstance(perf_or_path, dict):
        return perf_or_path
    if perf_or_path is None:
        return _read_json(PERF_PATH)
    return _read_json(Path(perf_or_path))


def _as_policy_obj(
    policy_or_thresholds: Union[Dict[str, Any], None],
    min_auroc: Optional[float],
    min_ks: Optional[float],
) -> Policy:
    if isinstance(policy_or_thresholds, dict):
        return Policy.from_dict(policy_or_thresholds)
    # fall back to explicit args
    return Policy(min_auroc=min_auroc, min_ks=min_ks)


def validate_policy(
    perf_or_path: Union[Dict[str, Any], str, Path, None] = PERF_PATH,
    policy_or_thresholds: Union[Dict[str, Any], None] = None,
    *,
    out_path: Union[str, Path] = OUT_PATH,
    min_auroc: Optional[float] = 0.70,
    min_ks: Optional[float] = 0.10,
) -> Dict[str, Any]:
    """
    Flexible validator.

    Examples:
      validate_policy({"auroc":0.9,"ks_stat":0.2}, {"min_auroc":0.7,"min_ks":0.1})
      validate_policy("reports/performance_metrics.json")  # uses defaults
    """
    _ensure_reports_dir()
    perf = _as_perf_dict(perf_or_path)
    policy = _as_policy_obj(policy_or_thresholds, min_auroc, min_ks)

    auroc = perf.get("auroc")
    ks = perf.get("ks_stat")

    reasons: List[str] = []
    status = "PASS"

    # AUROC
    if auroc is None or policy.min_auroc is None:
        reasons.append("AUROC missing")
        status = "FAIL"
    elif float(auroc) < float(policy.min_auroc):
        reasons.append(f"AUROC {auroc} < min_auroc {policy.min_auroc}")
        status = "FAIL"

    # KS
    if ks is None or policy.min_ks is None:
        reasons.append("KS missing")
        status = "FAIL"
    elif float(ks) < float(policy.min_ks):
        reasons.append(f"KS {ks} < min_ks {policy.min_ks}")
        status = "FAIL"

    result = {
        "status": status,
        "policy": policy.as_dict(),
        "reasons": reasons,
        "inputs": {
            "source": "dict" if isinstance(perf_or_path, dict) else str(perf_or_path)
        },
    }

    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    res = validate_policy()
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
