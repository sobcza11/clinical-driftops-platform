# tests/test_validate_cli.py
import json, subprocess, sys, os
from pathlib import Path

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)

def test_validate_cli_end_to_end(mini_workspace):
    # Run validator on the seeded predictions
    rc = _run([sys.executable, "src/api/validate_cli.py", "--preds", "reports/predictions.csv"])
    assert rc.returncode in (0, 1)  # gate may pass/fail; CLI itself should exit with gate code

    reports = mini_workspace["reports"]
    # Artifacts exist
    for p in [
        "performance_metrics.json",
        "performance_metrics.csv",
        "api_fairness_metrics.csv",
        "api_fairness_report.md",
        "fairness_summary.json",
        "policy_gate_result.json",
        "live_validation.json",
    ]:
        assert (reports / p).exists(), f"missing reports/{p}"

    # Live JSON shape (minimal)
    live = json.loads((reports / "live_validation.json").read_text(encoding="utf-8"))
    assert "status" in live and live["status"] in ("PASS", "FAIL")
    assert "performance" in live and "gate" in live
