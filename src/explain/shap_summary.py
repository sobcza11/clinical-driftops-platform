"""
Generate SHAP summary plot for top features.
Usage:
    python -m src.explain.shap_summary --data data/data_prepared_current.csv --out reports/shap_top_features.png --topk 15
"""

import argparse
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # important for GitHub Actions headless environment
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to prepared CSV")
    parser.add_argument("--out", required=True, help="Output path for SHAP summary plot")
    parser.add_argument("--topk", type=int, default=15, help="Top K features to show")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "label" not in df.columns:
        print("⚠️ No 'label' column – writing a stub plot.")
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No label column found", ha="center", va="center")
        plt.axis("off")
        plt.savefig(args.out, bbox_inches="tight")
        return

    # separate features/target
    y = df["label"]
    X = df.drop(columns=["label"])

    # encode non-numeric columns
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[1] if len(set(y)) > 1 else explainer.shap_values(X)

    shap.summary_plot(
        shap_values, X, plot_type="bar", max_display=args.topk, show=False
    )
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"✅ saved SHAP summary → {args.out}")

if __name__ == "__main__":
    main()