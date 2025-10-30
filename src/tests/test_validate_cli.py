import json, pathlib, subprocess, sys, os

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = sys.executable

def test_validate_cli_end_to_end(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "predictions.csv").write_text(
        "y_true,y_pred_prob\n1,0.91\n0,0.12\n1,0.77\n0,0.05\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    # force stable path (no external argparse modules)
    env["DRIFTOPS_ENABLE_EXTERNAL"] = "0"

    cmd = [PY, str(ROOT / "src" / "api" / "validate_cli.py"),
           "--preds", str(reports / "predictions.csv")]
    subprocess.check_call(cmd, cwd=ROOT, env=env)

    lv = json.loads((reports / "live_validation.json").read_text(encoding="utf-8"))
    assert lv["status"] in {"PASS", "FAIL"}
    assert "performance" in lv and "gate" in lv
    assert (reports / "policy_gate_result.json").exists()
    assert (reports / "performance_metrics.json").exists()
