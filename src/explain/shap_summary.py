"""
Phase V — SHAP Summary Generator (numeric-only features)
"""
import argparse
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--topk", type=int, default=15)
    args = p.parse_args()

    df = pd.read_csv(args.data)

    # separate label
    y = df["label"] if "label" in df.columns else None
    if y is None:
        print("⚠️ No 'label' column – writing a stub plot.")
        plt.figure()
        plt.text(0.5, 0.5, "No label column", ha="center", va="center")
        plt.axis("off")
        plt.savefig(args.out, bbox_inches="tight", dpi=160)
        return

    # keep only numeric features (drop IDs/time/categoricals implicitly)
    X = df.select_dtypes(include="number").drop(columns=["label"], errors="ignore")
    if X.shape[1] == 0:
        print("⚠️ No numeric features — writing stub plot.")
        plt.figure()
        plt.text(0.5, 0.5, "No numeric features", ha="center", va="center")
        plt.axis("off")
        plt.savefig(args.out, bbox_inches="tight", dpi=160)
        return

    # train simple model
    model = RandomForestClassifier(n_estimators=64, random_state=42)
    model.fit(X, y)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap.summary_plot(shap_values, X, show=False, max_display=args.topk)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight", dpi=160)
    print(f"✅ saved SHAP summary → {args.out}")

if __name__ == "__main__":
    main()