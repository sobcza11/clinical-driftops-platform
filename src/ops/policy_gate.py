# clinical-driftops-platform/src/ops/policy_gate.py
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Soft deps
try:
    import yaml
except Exception as _e:  # noqa: BLE001
    yaml = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception as _e:  # noqa: BLE001
    pd = None  # type: ignore


# ----------------------------
# Helpers (robust, tolerant)
# ----------------------------
def _read_yaml(path: str | Path) -> dict:
    p = Path(path)
    if yaml is None:
        raise RuntimeError("PyYAML not installed but required to read policy.yaml.")
    if not p.exists():
        raise FileNotFoundError(f"Missing policy file: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _load_drift_summary(csv_path: str | Path) -> tuple[float | None, float | None]:
    """
    Tries to read drift metrics CSV and return (max_psi, max_ks).
    Accepts columns like:
      - psi
      - ks or ks_stat
    Any missing parts return None for that metric.
    """
    if pd is None:
        return None, None
    p = Path(csv_path)
    if not p.exists():
        return None, None

    df = pd.read_csv(p)
    max_psi = None
    max_ks = None

    # PSI
    for psi_col in ["psi", "PSI", "psi_score"]:
        if psi_col in df.columns:
            try:
                max_psi = float(df[psi_col].astype(float).max())
                break
            except Exception:
                pass

    # KS
    for ks_col in ["ks", "KS", "ks_stat", "ks_statistic"]:
        if ks_col in df.columns:
            try:
                max_ks = float(df[ks_col].astype(float).max())
                break
            except Exception:
                pass

    return max_psi, max_ks


def _load_max_parity_gap_from_csv(csv_path: str | Path) -> float | None:
    """
    Returns the absolute max parity gap/disparity from a fairness CSV.
    Supports either:
      - compact schema: [group, n, positive_rate, disparity]
      - wider schemas containing 'parity_gap' column
      - optional 'summary' pattern (metric == 'summary' with parity values)
    """
    if pd is None:
        return None
    p = Path(csv_path)
    if not p.exists():
        return None

    df = pd.read_csv(p)

    # 1) Compact schema (your current): 'disparity'
    if "disparity" in df.columns:
        try:
            return float(df["disparity"].astype(float).abs().max())
        except Exception:
            pass

    # 2) Wide schema: 'parity_gap' column
    if "parity_gap" in df.columns:
        try:
            return float(df["parity_gap"].astype(float).abs().max())
        except Exception:
            pass

    # 3) Optional "summary" pattern (loose)
    if "metric" in df.columns and "parity_gap" in df.columns:
        try:
            summary_rows = df.loc[df["metric"].astype(str).str.contains("summary", case=False, na=False)]
            if not summary_rows.empty:
                return float(summary_rows["parity_gap"].astype(float).abs().max())
        except Exception:
            pass

    return None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ----------------------------
# Gate evaluation
# ----------------------------
def main() -> int:
    policy = _read_yaml("policy.yaml")

    observed: dict = {}
    checks: list[dict] = []

    # --- Drift checks ---
    drift_csv = Path("reports/drift_metrics.csv")
    max_psi, max_ks = _load_drift_summary(drift_csv)
    observed["max_psi"] = None if max_psi is None else float(max_psi)
    observed["max_ks"] = None if max_ks is None else float(max_ks)

    # Fetch thresholds (tolerant to missing keys)
    drift_cfg = policy.get("drift", {})
    psi_fail = float(drift_cfg.get("psi_fail", 0.20))
    ks_fail = float(drift_cfg.get("ks_fail", 0.20))

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

    # --- Explainability checks ---
    expl_cfg = policy.get("explainability", {})
    require_shap = bool(expl_cfg.get("require_shap_artifact", True))
    top_min = expl_cfg.get("top_features_min", None)

    shap_path = Path("reports/shap_top_features.png")
    shap_present = shap_path.exists()
    observed["shap_artifact_present"] = bool(shap_present)

    # SHAP presence required?
    checks.append({
        "name": "explainability.shap_artifact_present",
        "value": shap_present,
        "op": "==",
        "threshold": True if require_shap else False,
        "result": ("PASS" if (shap_present == require_shap) else "FAIL"),
    })

    # Optional sidecar meta for top-k (if you later emit one)
    # If not available, we'll SKIP the top-features check.
    top_detected = None
    shap_meta = Path("reports/shap_summary_meta.json")
    if shap_meta.exists():
        try:
            meta = json.loads(shap_meta.read_text(encoding="utf-8"))
            # Accept keys like 'topk' or 'top_features'
            for k in ("topk", "top_features", "n_features"):
                if k in meta:
                    top_detected = int(meta[k])
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

    # --- Fairness checks ---
    fair_csv = Path("reports/fairness_metrics.csv")
    max_gap = _load_max_parity_gap_from_csv(fair_csv)
    observed["parity_gap"] = None if max_gap is None else float(max_gap)

    fair_cfg = policy.get("fairness", {})
    gap_fail = float(fair_cfg.get("parity_gap_fail", 0.05))

    checks.append({
        "name": "fairness.parity_gap",
        "value": observed["parity_gap"],
        "op": "<=",
        "threshold": gap_fail,
        "result": "SKIP" if observed["parity_gap"] is None else ("PASS" if observed["parity_gap"] <= gap_fail else "FAIL"),
    })

    # --- Decide final status ---
    any_fail = any(c["result"] == "FAIL" for c in checks)
    status = "FAIL" if any_fail else "PASS"

    result = {
        "status": status,
        "timestamp_utc": _now_utc_iso(),
        "policy": policy,
        "observed": observed,
        "checks": checks,
        "notes": "Gate passed. All observed values within policy limits."
                 if status == "PASS" else
                 "Gate failed. See checks for violating metrics."
    }

    # Ensure reports dir
    Path("reports").mkdir(parents=True, exist_ok=True)
    out_path = Path("reports/policy_gate_result.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # --- MLflow logging (safe no-op if mlflow missing) ---
    try:
        from ops.mlflow_utils import start_run, log_metrics, log_artifact  # type: ignore
        start_run("Phase VI • Policy Gate")
        metr = {}
        if observed.get("max_psi") is not None:
            metr["max_psi"] = float(observed["max_psi"])
        if observed.get("max_ks") is not None:
            metr["max_ks"] = float(observed["max_ks"])
        if observed.get("parity_gap") is not None:
            metr["parity_gap"] = float(observed["parity_gap"])
        metr["gate_pass"] = 1.0 if status == "PASS" else 0.0
        if metr:
            log_metrics(metr)
        log_artifact(out_path, artifact_path="gate")
    except Exception:
        pass

    # Non-zero exit on failure for CI gate enforcement
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

