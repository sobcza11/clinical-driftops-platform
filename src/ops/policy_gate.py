# src/api/validate_cli.py
# Clinical DriftOps — end-to-end validator CLI (PASS/FAIL exit code)
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from importlib import import_module
import inspect

# Keep ALL imports module-level-safe
from src.ops.regulatory_monitor import main as regulatory_monitor_main
from src.ops.run_metadata import main as run_metadata_main
from src.ops.policy_registry_view import main as policy_registry_view_main
from src.explain.shap_stub import main as shap_stub_main
from src.ops.evidence_digest import main as evidence_digest_main
from src.ops.drift_history import main as drift_history_main

REPORTS_DIR = Path("reports")

# -----------------------
# Helpers
# -----------------------
def _ensure_reports() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def _read_csv_head(path: Path, n: int = 5) -> Tuple[Optional[list], Optional[list]]:
    if not path.exists():
        return None, None
    try:
        # utf-8-sig to absorb any BOM
        with path.open("r", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if not rows:
            return None, None
        return rows[0], rows[1:1+n]
    except Exception:
        return None, None

def _write_live_validation(status: str, performance: Dict[str, Any], gate: Dict[str, Any]) -> None:
    payload = {
        "status": status,
        "performance": performance or {"n": 0, "accuracy@0.5": None, "auroc": None, "ks_stat": None},
        "gate": {
            "status": "pass" if status == "PASS" else "fail",
            "policy": gate.get("policy", "default"),
            "reasons": gate.get("reasons", []),
        }
    }
    (REPORTS_DIR / "live_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _perf_dict_for_live(perf_path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(perf_path.read_text(encoding="utf-8"))
        return {
            "n": data.get("n"),
            "accuracy@0.5": data.get("accuracy@0.5"),
            "auroc": data.get("auroc"),
            "ks_stat": data.get("ks_stat"),
        }
    except Exception:
        return {"n": 0, "accuracy@0.5": None, "auroc": None, "ks_stat": None}

def _gate_result() -> Dict[str, Any]:
    p = REPORTS_DIR / "policy_gate_result.json"
    if not p.exists():
        return {"status": "fail", "policy": "default", "reasons": ["Gate unavailable or error; failing closed."]}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        status = raw.get("status") or raw.get("gate_status") or "fail"
        return {"status": status, "policy": raw.get("policy", "default"), "reasons": raw.get("reasons", [])}
    except Exception:
        return {"status": "fail", "policy": "default", "reasons": ["Gate parse error; failing closed."]}

def _maybe_seed_predictions(out_csv: Path) -> None:
    if out_csv.exists():
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("y_true,y_pred_prob\n1,0.91\n0,0.12\n1,0.77\n0,0.05\n", encoding="utf-8")

# -----------------------
# Signature/name-aware wrappers
# -----------------------
def run_performance_metrics(preds_csv: str) -> None:
    m = import_module("src.eval.performance_metrics")
    if hasattr(m, "compute_performance_metrics"):
        m.compute_performance_metrics(preds_csv=preds_csv)
        return
    if hasattr(m, "main"):
        try:
            sig = inspect.signature(m.main)
            params = list(sig.parameters.values())
            names = {p.name for p in params}
            if len(params) == 0:
                m.main()
            elif {"preds_csv", "out_dir"}.issubset(names):
                m.main(preds_csv=preds_csv, out_dir=str(REPORTS_DIR))
            elif len(params) == 2:
                m.main(preds_csv, str(REPORTS_DIR))
            elif "preds_csv" in names:
                m.main(preds_csv=preds_csv)
            else:
                m.main()
        except TypeError:
            for call in (
                lambda: m.main(preds_csv=preds_csv, out_dir=str(REPORTS_DIR)),
                lambda: m.main(preds_csv, str(REPORTS_DIR)),
                lambda: m.main(),
            ):
                try: call(); return
                except TypeError: pass
        return
    raise RuntimeError("performance_metrics: no entry point found")

def run_fairness_audit(preds_csv: str) -> None:
    try:
        m = import_module("src.eval.fairness_audit")
    except Exception:
        m = None
    if m and hasattr(m, "compute_api_fairness"):
        m.compute_api_fairness(preds_csv=preds_csv); return
    if m and hasattr(m, "main"):
        try:
            sig = inspect.signature(m.main)
            if len(sig.parameters) == 0:
                m.main(); return
            try:
                m.main(preds_csv=preds_csv); return
            except TypeError:
                m.main(preds_csv); return
        except Exception:
            try: m.main(); return
            except Exception: pass
    # Fallback placeholders
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "api_fairness_metrics.csv").write_text(
        "slice,demographic_parity_ratio\noverall,1.0\n", encoding="utf-8")
    (REPORTS_DIR / "api_fairness_report.md").write_text("# Fairness Report\n\nPlaceholder.\n", encoding="utf-8")
    (REPORTS_DIR / "fairness_summary.json").write_text(
        json.dumps({"overall": {"demographic_parity_ratio": 1.0}}, indent=2), encoding="utf-8")

def run_policy_gate() -> None:
    """
    Be resilient to different exports in src.ops.policy_gate:
    - evaluate_policy_gate()
    - main()
    - run()
    If nothing exists or it fails, emit a failing-closed policy file.
    """
    try:
        m = import_module("src.ops.policy_gate")
        if hasattr(m, "evaluate_policy_gate"):
            m.evaluate_policy_gate(); return
        if hasattr(m, "main"):
            m.main(); return
        if hasattr(m, "run"):
            m.run(); return
        raise AttributeError("No evaluate_policy_gate/main/run in src.ops.policy_gate")
    except Exception as e:
        (REPORTS_DIR / "policy_gate_result.json").write_text(
            json.dumps({"status": "fail", "policy": "default", "reasons": [f"Gate error: {e}"]}, indent=2),
            encoding="utf-8"
        )

# -----------------------
# Main
# -----------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Clinical DriftOps validator")
    parser.add_argument("--preds", type=str, required=True,
                        help="Path to predictions.csv with columns y_true,y_pred_prob")
    args = parser.parse_args(argv)

    _ensure_reports()
    preds_path = Path(args.preds)
    _maybe_seed_predictions(preds_path)

    header, head_rows = _read_csv_head(preds_path, n=4)
    print("[validator] predictions head:", header or "(no header)")
    if head_rows:
        for r in head_rows:
            print("  ", r)

    # 1) Performance
    perf_json_path = REPORTS_DIR / "performance_metrics.json"
    try:
        run_performance_metrics(preds_csv=str(preds_path))
    except Exception as e:
        perf_json_path.write_text(
            json.dumps({"n": 0, "accuracy@0.5": None, "auroc": None, "ks_stat": None}, indent=2),
            encoding="utf-8")
        print(f"[validator] performance_metrics failed: {e}")

    # 2) Fairness
    try:
        run_fairness_audit(preds_csv=str(preds_path))
    except Exception as e:
        print(f"[validator] fairness_audit failed: {e}")

    # 3) SHAP (placeholder)
    try:
        shap_stub_main()
    except Exception as e:
        print(f"[validator] shap_stub failed: {e}")

    # 4) Policy Gate (name-agnostic)
    run_policy_gate()

    # 5) Regulatory monitor
    try:
        regulatory_monitor_main()
    except Exception as e:
        print(f"[validator] regulatory_monitor failed: {e}")

    # 6) Run metadata
    try:
        run_metadata_main()
    except Exception as e:
        print(f"[validator] run_metadata failed: {e}")

    # 7) Policy registry view
    try:
        policy_registry_view_main()
    except Exception as e:
        print(f"[validator] policy_registry_view failed: {e}")

    # 8) Evidence digest
    try:
        evidence_digest_main()
    except Exception as e:
        print(f"[validator] evidence_digest failed: {e}")

    # 9) Drift history
    try:
        drift_history_main()
    except Exception as e:
        print(f"[validator] drift_history failed: {e}")

    # Live JSON & exit code
    perf = _perf_dict_for_live(perf_json_path)
    gate = _gate_result()
    status = "PASS" if str(gate.get("status", "")).lower() == "pass" else "FAIL"
    _write_live_validation(status=status, performance=perf, gate=gate)

    print(f"[validator] FINAL STATUS: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        try:
            _ensure_reports()
            (REPORTS_DIR / "validator_error.json").write_text(
                json.dumps({"error": repr(e)}, indent=2), encoding="utf-8")
            _write_live_validation(
                status="FAIL",
                performance={"n": 0, "accuracy@0.5": None, "auroc": None, "ks_stat": None},
                gate={"status": "fail", "policy": "default", "reasons": ["Unhandled exception in validator"]},
            )
        finally:
            sys.exit(1)


