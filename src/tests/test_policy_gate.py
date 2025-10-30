# tests/test_policy_gate.py
import json
from src.ops.policy_gate import main as gate_main

def test_policy_gate_pass(mini_workspace):
    # Minimal fairness summary (optional – validator usually writes it)
    (mini_workspace["reports"] / "fairness_summary.json").write_text(
        json.dumps({"parity_gap": 0.01}, indent=2), encoding="utf-8"
    )
    # Minimal performance metrics
    (mini_workspace["reports"] / "performance_metrics.json").write_text(
        json.dumps({"auroc": 1.0, "auprc": 1.0, "log_loss": 0.12}, indent=2), encoding="utf-8"
    )
    # Run gate
    rc = gate_main()
    out = json.loads((mini_workspace["reports"] / "policy_gate_result.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert out["status"] == "PASS"
    # sanity: drift should be finite & below fail thresholds
    assert out["observed"]["max_psi"] is None or out["observed"]["max_psi"] < 0.2
    assert out["observed"]["max_ks"]  is None or out["observed"]["max_ks"]  < 0.2
