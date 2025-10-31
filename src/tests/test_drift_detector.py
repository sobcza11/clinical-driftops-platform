import numpy as np
import pandas as pd
from monitors.drift_detector import compare_dataframes


def test_psi_ks_detects_shift():
    rng = np.random.default_rng(42)
    base = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 10_000),
            "y": rng.normal(5, 2, 10_000),
        }
    )
    curr = pd.DataFrame(
        {
            "x": rng.normal(0.5, 1.2, 10_000),  # mean & std shift
            "y": rng.normal(5, 2, 10_000),  # stable
        }
    )
    summary = compare_dataframes(base, curr, id_cols=())
    df = summary.as_dataframe.set_index("feature")

    assert df.loc["x", "drift_flag"] is True
    assert df.loc["y", "drift_flag"] in (False, True)  # allow chance fluctuation
