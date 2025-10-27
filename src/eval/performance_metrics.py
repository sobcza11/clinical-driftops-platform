# clinical-driftops-platform/src/eval/performance_metrics.py
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

# Soft deps
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
except Exception as _e:
    roc_auc_score = average_precision_score = accuracy_score = None  # type: ignore

POSSIBLE_FILES = [
    Path("reports/predictions.csv"),
    Path("reports/predictions_current.csv"),
    Path("data/predictions.csv"),
]

def find_predictions() -> Path | None:
    for p in POSSIBLE_FILES:
        if p.exists():
            return p
    return None

def load_preds(path: Path) -> pd.DataFrame:
    """
    Accepts columns:
      - y_true (0/1)
      - y_score (float prob)  [preferred]
      - y_pred (0/1)          [optional if no y_score]
    """
    df = pd.read_csv(path)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    # required target
    y_true_col = cols.get("y_true")
    if not y_true_col:
        # try label, target
        y_true_col = cols.get("label") or cols.get("target")
    if not y_true_col:
        raise ValueError("predictions CSV must include y_true (or label/target).")

    # score and/or pred
    y_score_col = cols.get("y_score") or cols.get("prob") or cols.get("proba") or cols.get("score")
    y_pred_col = cols.get("y_pred") or cols.get("pred")
    return df, y_true_col, y_score_col, y_pred_col

def main() -> int:
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    pred_path = find_predictions()
    out_json = reports / "performance_metrics.json"
    out_csv = reports / "performance_metrics.csv"

    result = {
        "status": "SKIP",
        "notes": "No predictions file found; looked for reports/predictions.csv, reports/predictions_current.csv, data/predictions.csv",
        "auroc": None,
        "auprc": None,
        "accuracy": None,
        "n_samples": 0,
        "source": None,
    }

    try:
        if pred_path is None:
            out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
            pd.DataFrame([result]).to_csv(out_csv, index=False)
            return 0

        df, y_true_col, y_score_col, y_pred_col = load_preds(pred_path)
        result["source"] = str(pred_path)
        result["n_samples"] = int(len(df))

        if roc_auc_score is None:
            result["status"] = "SKIP"
            result["notes"] = "scikit-learn not installed; cannot compute AUROC/AUPRC."
        else:
            y_true = df[y_true_col].astype(int)

            auroc = auprc = acc = None

            if y_score_col:
                y_score = df[y_score_col].astype(float)
                auroc = float(roc_auc_score(y_true, y_score))
                auprc = float(average_precision_score(y_true, y_score))
                # default threshold 0.5 for accuracy if y_pred missing
                y_pred = (y_score >= 0.5).astype(int)
                acc = float(accuracy_score(y_true, y_pred))
            elif y_pred_col:
                y_pred = df[y_pred_col].astype(int)
                acc = float(accuracy_score(y_true, y_pred))
                # AUROC/AUPRC require scores; leave None
            else:
                result["status"] = "SKIP"
                result["notes"] = "No y_score (prob) or y_pred found; cannot compute metrics."

            if auroc is not None or auprc is not None or acc is not None:
                result["status"] = "OK"
                result["notes"] = "Computed metrics."

            result["auroc"] = auroc
            result["auprc"] = auprc
            result["accuracy"] = acc

        out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        pd.DataFrame([result]).to_csv(out_csv, index=False)

    except Exception as e:
        result["status"] = "ERROR"
        result["notes"] = f"{type(e).__name__}: {e}"
        out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        pd.DataFrame([result]).to_csv(out_csv, index=False)
        return 0  # don't hard-fail CI; gate will enforce thresholds if present

    # --- MLflow logging (safe no-op if mlflow missing) ---
    try:
        from ops.mlflow_utils import start_run, log_metrics, log_artifact  # type: ignore
        start_run("Phase V • Performance")
        metr = {}
        if result.get("auroc") is not None:
            metr["auroc"] = float(result["auroc"])
        if result.get("auprc") is not None:
            metr["auprc"] = float(result["auprc"])
        if result.get("accuracy") is not None:
            metr["accuracy"] = float(result["accuracy"])
        if metr:
            log_metrics(metr)
        log_artifact(out_json, artifact_path="performance")
        log_artifact(out_csv, artifact_path="performance")
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

