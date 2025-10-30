# tests/test_dashboard.py
from pathlib import Path
import json, subprocess, sys

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)

def test_dashboard_builds(mini_workspace):
    # Ensure minimal required artifacts
    rep = mini_workspace["reports"]
    (rep / "performance_metrics.json").write_text(json.dumps({"auroc":1.0,"auprc":1.0,"log_loss":0.1}, indent=2), encoding="utf-8")
    (rep / "policy_gate_result.json").write_text(json.dumps({"status":"PASS","checks":[]}, indent=2), encoding="utf-8")
    (rep / "live_validation.json").write_text(json.dumps({"status":"PASS"}, indent=2), encoding="utf-8")
    (rep / "fairness_summary.json").write_text(json.dumps({"parity_gap":0.01}, indent=2), encoding="utf-8")
    (rep / "shap_top_features.json").write_text(json.dumps({"n_top_features":3,"features":["a","b","c"]}, indent=2), encoding="utf-8")

    # Build dashboard
    rc = _run([sys.executable, "src/reports_dashboard.py"])
    assert rc.returncode == 0, rc.stderr
    assert (rep / "index.html").exists()
    # Quick smoke: HTML file contains key sections
    html = (rep / "index.html").read_text(encoding="utf-8")
    assert "Clinical DriftOps — Reports Dashboard" in html
    assert "Policy Gate" in html
