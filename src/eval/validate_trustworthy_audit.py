"""
Validate Trustworthy AI audit results against policy.yaml thresholds.

Default behavior: soft validation (never breaks CI). To enforce strict mode,
set environment variable TRUST_AUDIT_STRICT=1 or pass --strict on CLI.

Expected files:
- policy.yaml
- reports/live_validation.json   (produced by your validation/metrics steps)

This script prints a compact summary and exits with:
  0 on success (or soft-missing artifacts when not strict)
  1 on policy violation (strict) or fatal errors in strict mode
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, NoReturn, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # Degrade gracefully if PyYAML not present


# ---------- paths ----------
POLICY_PATH = Path("policy.yaml")
LIVE_JSON = Path("reports/live_validation.json")


# ---------- utils ----------
def die(msg: str) -> NoReturn:
    print(f"[trustworthy_audit] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def warn(msg: str) -> None:
    print(f"[trustworthy_audit] WARN: {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[trustworthy_audit] {msg}")


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as e:
        die(f"Failed to parse JSON at {path}: {e}")


def load_policy(path: Path) -> Dict[str, Any]:
    if yaml is None:
        warn("PyYAML not installed; using permissive defaults.")
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise
    except Exception as e:
        die(f"Failed to parse YAML at {path}: {e}")


def getenv_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ---------- core checks ----------
def check_performance(metrics: Dict[str, Any], policy: Dict[str, Any]) -> list[str]:
    """Compare AUROC/AUPRC/log_loss (if present) to policy thresholds."""
    out: list[str] = []
    ppol = (policy.get("performance") or {}) if isinstance(policy, dict) else {}

    def get_num(d: Dict[str, Any], key: str) -> Optional[float]:
        v = d.get(key)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    auroc = get_num(metrics, "auroc")
    min_auroc = get_num(ppol, "min_auroc")
    if min_auroc is not None and auroc is not None and auroc < min_auroc:
        out.append(f"AUROC {auroc:.3f} < min_auroc {min_auroc:.3f}")

    auprc = get_num(metrics, "auprc")
    min_auprc = get_num(ppol, "min_auprc")
    if min_auprc is not None and auprc is not None and auprc < min_auprc:
        out.append(f"AUPRC {auprc:.3f} < min_auprc {min_auprc:.3f}")

    log_loss = get_num(metrics, "log_loss")
    max_log_loss = get_num(ppol, "max_log_loss")
    if max_log_loss is not None and log_loss is not None and log_loss > max_log_loss:
        out.append(f"log_loss {log_loss:.3f} > max_log_loss {max_log_loss:.3f}")

    return out


def check_drift(drift: Dict[str, Any], policy: Dict[str, Any]) -> list[str]:
    out: list[str] = []
    dpol = (policy.get("drift") or {}) if isinstance(policy, dict) else {}

    def get_num(d: Dict[str, Any], key: str) -> Optional[float]:
        v = d.get(key)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    psi = get_num(drift, "psi")
    ks = get_num(drift, "ks")
    psi_fail = get_num(dpol, "psi_fail")
    ks_fail = get_num(dpol, "ks_fail")

    if psi_fail is not None and psi is not None and psi > psi_fail:
        out.append(f"PSI {psi:.3f} > psi_fail {psi_fail:.3f}")
    if ks_fail is not None and ks is not None and ks > ks_fail:
        out.append(f"KS {ks:.3f} > ks_fail {ks_fail:.3f}")

    return out


def check_fairness(fair: Dict[str, Any], policy: Dict[str, Any]) -> list[str]:
    out: list[str] = []
    fpol = (policy.get("fairness") or {}) if isinstance(policy, dict) else {}

    def get_num(d: Dict[str, Any], key: str) -> Optional[float]:
        v = d.get(key)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    gap = get_num(fair, "parity_gap")
    gap_fail = get_num(fpol, "parity_gap_fail")
    if gap_fail is not None and gap is not None and abs(gap) > gap_fail:
        out.append(f"parity_gap {gap:.3f} exceeds {gap_fail:.3f}")

    return out


def check_explainability(explain: Dict[str, Any], policy: Dict[str, Any]) -> list[str]:
    out: list[str] = []
    xpol = (policy.get("explainability") or {}) if isinstance(policy, dict) else {}

    require_art = bool(xpol.get("require_shap_artifact", False))
    min_top = xpol.get("top_features_min")
    try:
        min_top = int(min_top) if min_top is not None else None
    except Exception:
        min_top = None

    has_art = bool(explain.get("shap_artifact_present", False))
    n_top = explain.get("top_features_count")
    try:
        n_top = int(n_top) if n_top is not None else None
    except Exception:
        n_top = None

    if require_art and not has_art:
        out.append("SHAP artifact required but missing")
    if min_top is not None and n_top is not None and n_top < min_top:
        out.append(f"top_features_count {n_top} < required {min_top}")

    return out


# ---------- main ----------
def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    strict = getenv_truthy("TRUST_AUDIT_STRICT") or ("--strict" in argv)

    # Load policy
    try:
        policy = load_policy(POLICY_PATH)
    except FileNotFoundError:
        msg = f"{POLICY_PATH} not found"
        if strict:
            die(msg)
        warn(msg + " (soft mode: skipping policy checks)")
        policy = {}

    # Load live validation json
    try:
        live = load_json(LIVE_JSON)
    except FileNotFoundError:
        msg = f"{LIVE_JSON} not found"
        if strict:
            die(msg)
        warn(msg + " (soft mode: skipping audit)")
        info("RESULT: SKIPPED (no live_validation.json)")
        return 0

    # Expected structure (tolerant to variants)
    metrics = live.get("metrics", {}) or live.get("performance", {}) or {}
    drift = live.get("drift", {}) or {}
    fairness = live.get("fairness", {}) or {}
    explain = live.get("explainability", {}) or {}

    violations: list[str] = []
    violations += check_performance(metrics, policy)
    violations += check_drift(drift, policy)
    violations += check_fairness(fairness, policy)
    violations += check_explainability(explain, policy)

    # Pretty print summary
    info("SUMMARY:")
    info(f"  performance: {json.dumps(metrics, ensure_ascii=False)}")
    info(f"  drift:       {json.dumps(drift, ensure_ascii=False)}")
    info(f"  fairness:    {json.dumps(fairness, ensure_ascii=False)}")
    info(f"  explain:     {json.dumps(explain, ensure_ascii=False)}")

    if violations:
        for v in violations:
            warn(f"Policy violation: {v}")
        if strict:
            info("RESULT: FAIL (strict)")
            return 1
        else:
            info("RESULT: SOFT-FAIL (non-blocking)")
            return 0

    info("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
