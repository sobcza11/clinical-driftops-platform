from explain.shap_summary import compute_top_features


def test_shap_handles_missing_inputs():
    assert compute_top_features(None, None).features == []


def test_shap_basic_array():
    import numpy as np

    sv = np.array([[1.0, -2.0], [0.5, 0.0]])
    names = ["f1", "f2"]
    out = compute_top_features(sv, names, topk=2).features
    assert {f["name"] for f in out} == {"f1", "f2"}
