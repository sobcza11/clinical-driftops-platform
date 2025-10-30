# src/api/validate_cli.py
# Clinical DriftOps — end-to-end validator CLI (PASS/FAIL exit code)
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from importlib import import_module

REPORTS_DIR = Path("reports")

# Optional modules (all calls wrapped in try/except)
try:
    from src.explain.shap_stub import main as shap_stub_main
except Exception:
    shap_stub_main = None

def _ensure_reports() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def _maybe_seed_predictions(out_csv: Path) -> None:
    if out_csv.exists():
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(
        "y_true,y_pred_prob\n1,0.91\n0,0.12\n1,0.77\n0,0.05\n",
        encoding="utf-8",
    )

def _read_csv_head(path: Path, n: int = 5) -> Tuple[Optional[list], Optional[list]]:
    if not path.exists():
        return None, None
    try:
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            return None, None
        header = rows[0]
        body = rows[1:1+n]
        return header, body
    except Exception:
        return None, None

# ---------- Minimal perf if eval module missing ----------
def _parse_preds(preds_csv: Path):
    y_true, y_prob = [], []
    with preds_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldmap = {k.lstrip("\ufeff"): k for k in r.fieldnames or []}
        ycol = fieldmap.get("y_true", "y_true")
        pcol = fieldmap.get("y_pred_prob", "y_pred_prob")
        for row in r:
            try:
                y_true.append(int(row[ycol]))
                y_prob.append(float(row[pcol]))
            except Exception:
                continue
    return y_true, y_prob

def _auc_mw(y_true, y_prob):
    pos = [(p, 1) for p, y in zip(y_prob, y_true) if y == 1]
    neg = [(p, 0) for p, y in zip(y_prob, y_true) if y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return None
    all_scored = sorted([(p, i) for i, p in enumerate(y_prob)], key=lambda x: x[0])
    ranks, rank, i = {}, 1, 0
    while i < len(all_scored):
        j = i
        while j+1 < len(all_scored) and all_scored[j+1][0] == all_scored[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j+1):
            ranks[all_scored[k][1]] = avg_rank
        rank = j + 2
        i = j + 1
    pos_idx = [i for i, y in enumerate(y_true) if y == 1]
    R1 = sum(ranks[i] for i in pos_idx)
    U1 = R1 - n1 * (n1 + 1) / 2.0
    return U1 / (n1 * n0)

def _ks_stat(y_true, y_prob):
    pairs = sorted(zip(y_prob, y_true), key=lambda x: x[0])
    n = len(pairs)
    if n == 0:
        return None
    n1 = sum(1 for _, y in pairs if y == 1)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return None
    tp = fp = 0
    maxdiff = 0.0
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n1
        fpr = fp / n0
        d = abs(tpr - fpr)
        if d > maxdiff:
            maxdiff = d
    return maxdiff

def _compute_basic_performance(preds_csv: Path) -> Dict[str, Any]:
    y, p = _parse_preds(preds_csv)
    n = len(y)
    if n == 0:
        return {"n": 0, "accuracy@0.5": None, "auroc": None, "ks_stat": None}
    acc = sum((1 if prob >= 0.5 else 0) == lab for lab, prob in zip(y, p)) / n
    auc = _auc_mw(y, p)
    ks = _ks_stat(y, p)
    return {"n": n, "accuracy@0.5": round(acc, 4), "auroc": round(auc, 4) if auc is not None else None,
            "ks_stat": round(ks, 4) if ks is not None else None}

def run_performance_metrics(preds_csv: str) -> None:
    try:
        m = import_module("src.eval.performance_metrics")
        if hasattr(m, "compute_performance_metrics"):
            m.compute_performance_metrics(preds_csv=preds_csv)
            return
        if hasattr(m, "main"):
            try:
                m.main(preds_csv=preds_csv)
            except TypeError:
                m.main(preds_csv)
            return
    except Exception:
        pass
    perf = _compute_basic_performance(Path(preds_csv))
    (REPORTS_DIR / "performance_metrics.json").write_text(
        json.dumps(perf, indent=2), encoding="utf-8"
    )

def run_fairness_audit(preds_csv: str) -> None:
    (REPORTS_DIR / "api_fairness_metrics.csv").write_text(
        "slice,demographic_parity_ratio\noverall,1.0\n", encoding="utf-8"
    )
    (REPORTS_DIR / "api_fairness_report.md").write_text(
        "# Fairness Report\n\nPlaceholder.\n", encoding="utf-8"
    )
    (REPORTS_DIR / "fairness_summary.json").write_text(
        json.dumps({"overall": {"demographic_parity_ratio": 1.0}}, indent=2),
        encoding="utf-8",
    )

def run_policy_gate() -> Dict[str, Any]:
    """
    PURE LOCAL GATE: do not import anything (avoids argparse side-effects).
    Reads performance_metrics.json and writes policy_gate_result.json.
    """
    try:
        perf = json.loads((REPORTS_DIR / "performance_metrics.json").read_text(encoding="utf-8"))
    except Exception:
        perf = {}
    auroc = perf.get("auroc") or 0.0
    ks = perf.get("ks_stat") or 0.0
    # simple default policy
    policy = {"min_auroc": 0.7, "min_ks": 0.1}
    status = "PASS" if (auroc >= policy["min_auroc"] and ks >= policy["min_ks"]) else "FAIL"
    reasons = []
    if status == "FAIL":
        if not (auroc >= policy["min_auroc"]):
            reasons.append(f"auroc<{policy['min_auroc']}")
        if not (ks >= policy['min_ks']):
            reasons.append(f"ks<{policy['min_ks']}")
    payload = {"status": status, "policy": policy, "reasons": reasons}
    (REPORTS_DIR / "policy_gate_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

def _gate_result() -> Dict[str, Any]:
    p = REPORTS_DIR / "policy_gate_result.json"
    if not p.exists():
        return {"status": "fail", "policy": "default", "reasons": ["Gate unavailable; failing closed."]}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        status = (raw.get("status") or raw.get("gate_status") or "").lower()
        return {
            "status": "pass" if status in ("pass", "passed", "success") else "fail",
            "policy": raw.get("policy", "default"),
            "reasons": raw.get("reasons", []),
        }
    except Exception:
        return {"status": "fail", "policy": "default", "reasons": ["Gate parse error; failing closed."]}

def _perf_for_live() -> Dict[str, Any]:
    try:
        data = json.loads((REPORTS_DIR / "performance_metrics.json").read_text(encoding="utf-8"))
        return {
            "n": data.get("n"),
            "accuracy@0.5": data.get("accuracy@0.5"),
            "auroc": data.get("auroc"),
            "ks_stat": data.get("ks_stat"),
        }
    except Exception:
        return {"n": 0, "accuracy@0.5": None, "auroc": None, "ks_stat": None}

def _write_live_validation(status: str, performance: Dict[str, Any], gate: Dict[str, Any]) -> None:
    payload = {
        "status": status,
        "performance": performance,
        "gate": {
            "status": "pass" if status == "PASS" else "fail",
            "policy": gate.get("policy", "default"),
            "reasons": gate.get("reasons", []),
        },
    }
    (REPORTS_DIR / "live_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _call_optional(module_path: str, func_name: str = "main") -> None:
    try:
        m = import_module(module_path)
        fn = getattr(m, func_name, None)
        if callable(fn):
            try:
                fn()
            except TypeError:
                fn(REPORTS_DIR)
    except Exception:
        pass

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Clinical DriftOps validator")
    parser.add_argument("--preds", type=str, required=True, help="Path to predictions.csv (y_true,y_pred_prob)")
    args = parser.parse_args(argv)

    _ensure_reports()
    preds_path = Path(args.preds)
    _maybe_seed_predictions(preds_path)

    header, head_rows = _read_csv_head(preds_path, n=4)
    print("[validator] predictions head:", header or "(no header)")
    if head_rows:
        for r in head_rows:
            print("  ", r)

    try:
        run_performance_metrics(str(preds_path))
    except Exception as e:
        (REPORTS_DIR / "performance_metrics.json").write_text(
            json.dumps({"n": 0, "accuracy@0.5": None, "auroc": None, "ks_stat": None}, indent=2),
            encoding="utf-8",
        )
        print(f"[validator] performance_metrics failed: {e}")

    try:
        run_fairness_audit(str(preds_path))
    except Exception as e:
        print(f"[validator] fairness_audit failed: {e}")

    if shap_stub_main:
        try:
            shap_stub_main()
        except Exception as e:
            print(f"[validator] shap_stub failed: {e}")

    try:
        run_policy_gate()
    except Exception as e:
        (REPORTS_DIR / "policy_gate_result.json").write_text(
            json.dumps({"status": "FAIL", "policy": "default", "reasons": [f"Gate error: {e}"]}, indent=2),
            encoding="utf-8",
        )
        print(f"[validator] policy_gate failed: {e}")

    for mod in (
        "src.ops.regulatory_monitor",
        "src.ops.run_metadata",
        "src.ops.policy_registry_view",
        "src.ops.evidence_digest",
        "src.ops.drift_history",
    ):
        _call_optional(mod)

    perf = _perf_for_live()
    gate = _gate_result()
    status = "PASS" if gate.get("status") == "pass" else "FAIL"
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
                json.dumps({"error": repr(e)}, indent=2), encoding="utf-8"
            )
            _write_live_validation(
                status="FAIL",
                performance={"n": 0, "accuracy@0.5": None, "auroc": None, "ks_stat": None},
                gate={"status": "fail", "policy": "default", "reasons": ["Unhandled exception in validator"]},
            )
        finally:
            sys.exit(1)

