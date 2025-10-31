# src/ops/policy_validator.py
# Purpose: Validate a simple policy gate and (optionally) log to MLflow.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(p: Path, data: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def validate_policy(
    perf: Dict[str, Any],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a gate result dict with PASS/FAIL and reasons."""
    min_auroc = _safe_float(policy.get("min_auroc"))
    min_ks = _safe_float(policy.get("min_ks"))

    auroc = _safe_float(perf.get("auroc"))
    ks = _safe_float(perf.get("ks_stat"))

    reasons = []

    if min_auroc is not None and auroc is not None and auroc < min_auroc:
        reasons.append(f"AUROC {auroc} < min_auroc {min_auroc}")
    if min_ks is not None and ks is not None and ks < min_ks:
        reasons.append(f"KS {ks} < min_ks {min_ks}")

    status = "PASS" if not reasons else "FAIL"

    return {
        "status": status,
        "policy": {"min_auroc": min_auroc, "min_ks": min_ks},
        "actuals": {"auroc": auroc, "ks_stat": ks},
        "reasons": reasons,
    }


def main(
    reports_dir: Path = Path("reports"),
    perf_name: str = "performance_metrics.json",
    policy_name: str = "active_policy.json",
    out_name: str = "policy_gate_result.json",
    log_mlflow: bool = False,
) -> Path:
    """Read inputs from reports/, write gate result back to reports/."""
    perf = _read_json(reports_dir / perf_name)
    policy = _read_json(reports_dir / policy_name)

    # Default thresholds if not present
    if not policy:
        policy = {"min_auroc": 0.7, "min_ks": 0.1}

    result = validate_policy(perf, policy)
    out_path = reports_dir / out_name
    _write_json(out_path, result)

    if log_mlflow:
        try:
            import mlflow  # optional

            with mlflow.start_run(run_name="Phase VI • Policy Validation"):
                mlflow.log_params(
                    {
                        "min_auroc": result["policy"]["min_auroc"],
                        "min_ks": result["policy"]["min_ks"],
                    }
                )
                mlflow.log_metrics(
                    {
                        "auroc": result["actuals"].get("auroc") or 0.0,
                        "ks_stat": result["actuals"].get("ks_stat") or 0.0,
                    }
                )
                mlflow.set_tag("policy_status", result["status"])
        except Exception:
            # Best-effort logging; ignore MLflow errors in CI
            pass

    return out_path


if __name__ == "__main__":
    print(main())
