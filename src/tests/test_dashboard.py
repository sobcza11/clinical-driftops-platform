import pathlib
import subprocess
import sys
import os

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = sys.executable


def test_dashboard_builds(tmp_path):
    # Ensure required inputs exist
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "performance_metrics.json").write_text(
        '{"n":4,"accuracy@0.5":1.0,"auroc":1.0,"ks_stat":1.0}', encoding="utf-8"
    )
    (reports / "policy_gate_result.json").write_text(
        '{"status":"PASS","policy":{"min_auroc":0.7,"min_ks":0.1},"reasons":[]}',
        encoding="utf-8",
    )
    (reports / "fairness_summary.json").write_text(
        '{"overall":{"demographic_parity_ratio":1.0}}', encoding="utf-8"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    # run dashboard
    subprocess.check_call(
        [PY, str(ROOT / "src" / "reports_dashboard.py")], cwd=ROOT, env=env
    )

    assert (ROOT / "reports" / "index.html").exists()
