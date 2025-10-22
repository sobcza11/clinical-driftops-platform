import pandas as pd

def add_label(path):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        # demo rule: high lactate -> positive class
        df["label"] = (df.get("lactate", 0).fillna(0) >= 2.0).astype(int)
        df.to_csv(path, index=False)
        print(f"added label to {path}: pos_rate={df['label'].mean():.3f}")
    else:
        print(f"{path} already has label")

for p in ["data/data_prepared_current.csv","data/data_prepared_baseline.csv"]:
    add_label(p)
