import pandas as pd
import numpy as np
from src.eval.fairness_audit import audit_fairness


def test_fairness_runs_and_writes(tmp_path):
    # synthetic minimal data
    n = 400
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(1, 2, n),
            "gender": rng.choice(["M", "F"], size=n),
        }
    )
    # label depends slightly on x1
    df["label"] = (df["x1"] + rng.normal(0, 0.5, n) > 0).astype(int)

    data_path = tmp_path / "toy.csv"
    df.to_csv(data_path, index=False)

    out_csv = tmp_path / "fairness.csv"
    out_md = tmp_path / "fairness.md"

    audit_fairness(str(data_path), "gender", str(out_csv), str(out_md))

    assert out_csv.exists()
    assert out_md.exists()
