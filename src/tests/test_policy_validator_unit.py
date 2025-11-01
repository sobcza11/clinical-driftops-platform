from src.ops.policy_validator import validate_policy


def test_policy_pass():
    perf = {"auroc": 0.9, "ks_stat": 0.2}
    policy = {"min_auroc": 0.7, "min_ks": 0.1}
    res = validate_policy(perf, policy)
    assert res["status"] == "PASS"
    assert not res["reasons"]


def test_policy_fail():
    perf = {"auroc": 0.6, "ks_stat": 0.05}
    policy = {"min_auroc": 0.7, "min_ks": 0.1}
    res = validate_policy(perf, policy)
    assert res["status"] == "FAIL"
    assert any("AUROC" in r for r in res["reasons"])
    assert any("KS" in r for r in res["reasons"])
