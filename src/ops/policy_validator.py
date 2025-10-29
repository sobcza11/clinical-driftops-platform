# clinical-driftops-platform/src/ops/policy_validator.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

import yaml  # PyYAML
try:
    from jsonschema import validate, Draft202012Validator
except Exception as _e:
    Draft202012Validator = None  # type: ignore

# Soft MLflow logging (no-op if missing)
try:
    from .mlflow_utils import start_run, log_metrics, log_params, log_artifact  # type: ignore
except Exception:
    def start_run(*_, **__): return None
    def log_metrics(*_, **__): ...
    def log_params(*_, **__): ...
    def log_artifact(*_, **__): ...

POLICY_PATH = Path("policy.yaml")
REGISTRY_PATH = Path("policy_registry.yaml")
REPORT_PATH = Path("reports/policy_validation_report.json")

# --- JSON-Schema (kept here for simplicity; move to a separate file later if you’d like)
POLICY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Clinical DriftOps Policy Schema v1",
    "type": "object",
    "properties": {
        "drift": {
            "type": "object",
            "properties": {
                "psi_fail": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "ks_fail":  {"type": "number", "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["psi_fail", "ks_fail"]
        },
        "performance": {
            "type": "object",
            "properties": {
                "min_auroc":    {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "min_auprc":    {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "max_log_loss": {"type": "number", "minimum": 0.0}
            },
            "required": []
        },
        "fairness": {
            "type": "object",
            "properties": {
                "parity_gap_fail": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["parity_gap_fail"]
        },
        "explainability": {
            "type": "object",
            "properties": {
                "require_shap_artifact": {"type": "boolean"},
                "top_features_min":      {"type": "integer", "minimum": 1}
            },
            "required": ["require_shap_artifact"]
        }
    },
    "required": ["drift", "fairness", "explainability"],
    "additionalProperties": True
}

def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _read_yaml(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"Missing YAML: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def validate_policy() -> dict:
    policy = _read_yaml(POLICY_PATH)
    registry = _read_yaml(REGISTRY_PATH) if REGISTRY_PATH.exists() else {}
    issues: list[str] = []

    # JSON-Schema structural validation
    schema_ok = True
    if Draft202012Validator is None:
        schema_ok = False
        issues.append("jsonschema not installed; skipped structural validation.")
    else:
        v = Draft202012Validator(POLICY_SCHEMA)
        errors = sorted(v.iter_errors(policy), key=lambda e: e.path)
        if errors:
            schema_ok = False
            for e in errors:
                loc = ".".join(str(x) for x in e.path) or "<root>"
                issues.append(f"Schema error at {loc}: {e.message}")

    # Semantic sanity checks (bound relationships)
    sem_ok = True
    drift = policy.get("drift", {})
    perf  = policy.get("performance", {})
    fair  = policy.get("fairness", {})
    expl  = policy.get("explainability", {})

    # Examples of semantic rules
    if "psi_fail" in drift and drift["psi_fail"] > 0.5:
        sem_ok = False
        issues.append("drift.psi_fail > 0.5 is unusually lenient for clinical drift.")
    if "ks_fail" in drift and drift["ks_fail"] > 0.5:
        sem_ok = False
        issues.append("drift.ks_fail > 0.5 is unusually lenient for clinical drift.")
    if "min_auroc" in perf and perf["min_auroc"] < 0.6:
        sem_ok = False
        issues.append("performance.min_auroc < 0.60 may be too low for safety-critical use.")
    if "parity_gap_fail" in fair and fair["parity_gap_fail"] > 0.2:
        sem_ok = False
        issues.append("fairness.parity_gap_fail > 0.20 may allow large disparities.")
    if expl.get("require_shap_artifact") is False:
        issues.append("explainability.require_shap_artifact is False (allowed, but not recommended).")

    status = "PASS" if (schema_ok and sem_ok) else "WARN" if schema_ok else "FAIL"

    result = {
        "timestamp_utc": _utc(),
        "status": status,
        "schema_validation": "OK" if schema_ok else "NOT_RUN_OR_FAILED",
        "semantic_checks": "OK" if sem_ok else "HAS_ISSUES",
        "issues": issues,
        "policy_path": str(POLICY_PATH),
        "registry_path": str(REGISTRY_PATH) if REGISTRY_PATH.exists() else None,
        "registry_info": registry.get("policies") if registry else None,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Log to MLflow (best-effort)
    run_id = start_run("Phase VI • Policy Validation")
    log_params({
        "policy_file": str(POLICY_PATH),
        "registry_file": str(REGISTRY_PATH) if REGISTRY_PATH.exists() else "",
    })
    log_metrics({
        "policy_schema_ok": 1.0 if schema_ok else 0.0,
        "policy_semantic_ok": 1.0 if sem_ok else 0.0,
        "policy_status_pass": 1.0 if status == "PASS" else 0.0,
    })
    log_artifact(REPORT_PATH, artifact_path="policy_validation")

    return result

if __name__ == "__main__":
    res = validate_policy()
    print(json.dumps(res, indent=2))
    # Non-zero exit only on structural FAIL (schema issues break CI)
    if res["status"] == "FAIL":
        raise SystemExit(2)