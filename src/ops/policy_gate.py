# clinical-driftops-platform/src/ops/policy_gate.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Optional deps; gate should still run without pandas/mlflow
try:
    import yaml  # PyYAML
except Exception:
    yaml = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore


# ----------------------------
# Utilities
# ----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _read_yaml(path: str | Path) -> dict:
    p = Path(path)
    if yaml is None:
        raise RuntimeError("PyYAML is required to read policy.yaml (pip install PyYAML).")
    if not p.exists():
        raise FileNotFoundError(f"Missing policy file: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data or {}

def _read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _load_max_from_csv(csv_path: str | Path, candidates: list[str]) -> float | None:
    """Return max of the first existing numeric column in candidates; None if file/column missing."""
    if pd is None:
        return None
    p = Path(csv_path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    for col in candidates:
        if col in df.columns:
            try:
                return float(pd.to_numeric(df[col], errors="coerce").max())
            except Exception:
                pass
    return None

def _exists(path: str | Path) -> bool:
    return Path(path).exists()


# ----------------------------
# Gate logic
# ----------------------------
def main() -> int:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    policy = _read_yaml("policy.yaml")

    # Observed values we’ll capture for the report
    observed: dict[str, float | int | bool | None] = {}
    checks: list[dict] = []

    # ---- Drift (from reports/drift_metrics.csv)
    # Accept column names: PSI: ["psi", "PSI", "psi_score"]; KS: ["ks", "KS", "ks_stat", "ks_statistic"]
    drift_cfg = policy.get("drift", {}) or {}
    psi_fail = float(drift_cfg.get("psi_fail", 0.20))
    ks_fail = float(drift_cfg.get("ks_fail", 0.20))

    max_psi = _load_max_from_csv("reports/drift_metrics.csv", ["psi", "PSI", "psi_score"])
    max_ks  = _load_max_from_csv("reports/drift_metrics.csv", ["ks", "KS", "ks_stat", "ks_statistic"])

    observed["max_psi"] = None if max_psi is None else float(max_psi)
    observed["max_ks"]  = None if max_ks  is None else float(max_ks)

    checks.append({
        "name": "drift.psi",
        "value": observed["max_psi"],
        "op": "<",
        "threshold": psi_fail,
        "result": "SKIP" if observed["max_psi"] is None else ("PASS" if observed["max_psi"] < psi_fail else "FAIL"),
    })
    checks.append({
        "name": "drift.ks",
        "value": observed["max_ks"],
        "op": "<",
        "threshold": ks_fail,
        "result": "SKIP" if observed["max_ks"] is None else ("PASS" if observed["max_ks"] < ks_fail else "FAIL"),
    })

    # ---- Performance (from reports/performance_metrics.json)
    perf_cfg = policy.get("performance", {}) or {}
    min_auroc   = perf_cfg.get("min_auroc")     # e.g., 0.80
    min_auprc   = perf_cfg.get("min_auprc")     # e.g., 0.50
    max_logloss = perf_cfg.get("max_log_loss")  # e.g., 0.70 (note key name)

    perf = _read_json("reports/performance_metrics.json")
    auroc = perf.get("auroc")
    auprc = perf.get("auprc")
    log_loss_val = perf.get("log_loss")

    observed["auroc"] = auroc
    observed["auprc"] = auprc
    observed["log_loss"] = log_loss_val

    if min_auroc is not None:
        checks.append({
            "name": "performance.auroc",
            "value": auroc,
            "op": ">=",
            "threshold": float(min_auroc),
            "result": "SKIP" if auroc is None or (isinstance(auroc, float) and auroc != auroc)
                      else ("PASS" if float(auroc) >= float(min_auroc) else "FAIL"),
        })
    if min_auprc is not None:
        checks.append({
            "name": "performance.auprc",
            "value": auprc,
            "op": ">=",
            "threshold": float(min_auprc),
            "result": "SKIP" if auprc is None or (isinstance(auprc, float) and auprc != auprc)
                      else ("PASS" if float(auprc) >= float(min_auprc) else "FAIL"),
        })
    if max_logloss is not None:
        checks.append({
            "name": "performance.log_loss",
            "value": log_loss_val,
            "op": "<=",
            "threshold": float(max_logloss),
            "result": "SKIP" if log_loss_val is None or (isinstance(log_loss_val, float) and log_loss_val != log_loss_val)
                      else ("PASS" if float(log_loss_val) <= float(max_logloss) else "FAIL"),
        })

    # ---- Fairness (from reports/fairness_metrics.csv)
    # We take the absolute max of 'disparity' or 'parity_gap' if present.
    fair_cfg = policy.get("fairness", {}) or {}
    gap_fail = float(fair_cfg.get("parity_gap_fail", 0.05))

    max_gap = None
    if pd is not None and Path("reports/fairness_metrics.csv").exists():
        try:
            df_f = pd.read_csv("reports/fairness_metrics.csv")
            if "disparity" in df_f.columns:
                max_gap = float(pd.to_numeric(df_f["disparity"], errors="coerce").abs().max())
            elif "parity_gap" in df_f.columns:
                max_gap = float(pd.to_numeric(df_f["parity_gap"], errors="coerce").abs().max())
        except Exception:
            max_gap = None

    observed["parity_gap"] = None if max_gap is None else float(max_gap)
    checks.append({
        "name": "fairness.parity_gap",
        "value": observed["parity_gap"],
        "op": "<=",
        "threshold": gap_fail,
        "result": "SKIP" if observed["parity_gap"] is None else ("PASS" if observed["parity_gap"] <= gap_fail else "FAIL"),
    })

    # ---- Explainability (SHAP presence & optional top-k)
    expl_cfg = policy.get("explainability", {}) or {}
    require_shap = bool(expl_cfg.get("require_shap_artifact", True))
    top_min = expl_cfg.get("top_features_min")  # optional

    shap_png = _exists("reports/shap_top_features.png")
    observed["shap_artifact_present"] = shap_png
    checks.append({
        "name": "explainability.shap_artifact_present",
        "value": shap_png,
        "op": "==",
        "threshold": True if require_shap else False,
        "result": "PASS" if (shap_png == require_shap) else "FAIL",
    })

    # Optional: detect top-k from sidecar JSON if you emit it
    top_detected = None
    shap_meta = _read_json("reports/shap_summary_meta.json")
    for k in ("topk", "top_features", "n_features"):
        if k in shap_meta and shap_meta[k] is not None:
            try:
                top_detected = int(shap_meta[k])
                break
            except Exception:
                pass
    observed["top_features_detected"] = top_detected

    if top_min is not None:
        checks.append({
            "name": "explainability.top_features_min",
            "value": top_detected,
            "op": ">=",
            "threshold": int(top_min),
            "result": "SKIP" if top_detected is None else ("PASS" if int(top_detected) >= int(top_min) else "FAIL"),
        })

    # ---- Finalize
    any_fail = any(c["result"] == "FAIL" for c in checks)
    status = "FAIL" if any_fail else "PASS"

    result = {
        "status": status,
        "timestamp_utc": _utc_now_iso(),
        "policy": policy,
        "observed": observed,
        "checks": checks,
        "notes": (
            "Gate failed. See checks for violating metrics."
            if status == "FAIL"
            else "Gate passed. All observed values within policy limits."
        ),
    }

    out_path = reports_dir / "policy_gate_result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Best-effort MLflow logging (safe no-op if unavailable)
    try:
        import mlflow  # type: ignore
        mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
        with mlflow.start_run(run_name="Phase V • Policy Gate", nested=True):
            to_log = {}
            for k in ("max_psi", "max_ks", "parity_gap", "auroc", "auprc", "log_loss"):
                if observed.get(k) is not None:
                    to_log[k] = float(observed[k])  # type: ignore[arg-type]
            to_log["gate_pass"] = 1.0 if status == "PASS" else 0.0
            if to_log:
                mlflow.log_metrics(to_log)
            mlflow.log_artifact(str(out_path), artifact_path="gate")
    except Exception:
        pass

    # CI gate: non-zero exit on failure
    if status == "FAIL":
        print("Policy gate FAILED:")
        for c in checks:
            if c["result"] == "FAIL":
                print(f" - {c['name']}: value={c['value']} {c['op']} {c['threshold']}")
        return 1

    print("Policy gate PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())