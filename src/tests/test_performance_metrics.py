# tests/test_performance_metrics.py
import numpy as np
from src.eval.performance_metrics import compute_metrics


def test_compute_metrics_perfect_classifier():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.01, 0.99, 0.02, 0.98])
    m = compute_metrics(y_true, y_score)
    assert m["n_samples"] == 4
    assert m["positive_rate"] == 0.5
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0
    assert 0 < m["log_loss"] < 0.3
    assert m["f1@0.5"] == 1.0
    assert 0.0 <= m["best_f1_threshold"] <= 1.0
