import pandas as pd

for p in ["data/data_prepared_current.csv", "data/data_prepared_baseline.csv"]:
    try:
        df = pd.read_csv(p, nrows=1)
        print(p, "->", list(df.columns)[:20])
    except Exception as e:
        print(p, "ERR:", e)
