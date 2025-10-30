# src/eval/performance_metrics.py
# Purpose: Compute core performance metrics from a predictions CSV and write JSON/CSV artifacts.
# Contract:
#   - main(preds_path: str, out_dir: str) -> dict
#   - Writes {out_dir}/performance_metrics.json and {out_dir}/performance_metrics.csv (UTF-8, LF)

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Accept multiple common aliases to be resilient to upstream schema changes
PROB_ALIASES = {
    "y_pred_prob", "pred_prob", "prob", "score", "prediction", "y_proba",
    "proba", "p_hat", "p", "predicted_probability",
}
LABEL_ALIASES = {"y_true", "label", "target", "y", "y_actual", "actual", "truth"}

def _read_predictions(preds_path: Path) -> Tuple[List[Optional[float]], List[float]]:
    """
    Reads a predictions CSV with headers and tries to locate:
      - a probability/score column (float, required)
      - an optional binary label column {0,1}
    """
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file does not exist: {preds_path}")

    with preds_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = {h.strip(): h for h in (reader.fieldnames or [])}

        # Find columns by alias (case-insensitive)
        def _match(colset: set[str]) -> Optional[str]:
            for raw in headers:
                if raw.strip().lower() in colset:
                    return headers[raw]
            return None

        prob_col = _match({a.lower() for a in PROB_ALIASES})
        if not prob_col:
            raise ValueError(
                f"Missing probability column; expected one of {sorted(PROB_ALIASES)}; "
                f"got headers: {list(headers)}"
            )
        label_col = _match({a.lower() for a in LABEL_ALIASES})

        y_true: List[Optional[float]] = []
        y_prob: List[float] = []
        for row in reader:
            try:
                p = float(row[prob_col])
            except Exception:
                # Skip rows with unparsable probability
                continue
            y_prob.append(p)

            if label_col:
                v = row.get(label_col, None)
                if v is None or v == "":
                    y_true.append(None)
                else:
                    try:
                        y_true.append(float(v))
                    except Exception:
                        y_true.append(None)
            else:
                y_true.append(None)

    return y_true, y_prob


def _accuracy(y_true: List[Optional[float]], y_prob: List[float], threshold: float = 0.5) -> Optional[float]:
    labeled = [(yt, yp) for yt, yp in zip(y_true, y_prob) if yt is not None]
    if not labeled:
        return None
    correct = sum(1.0 for yt, yp in labeled if (yp >= threshold) == (yt >= 0.5))
    return correct / len(labeled)


def _roc_auc(y_true: List[Optional[float]], y_prob: List[float]) -> Optional[float]:
    """Label-free AUROC via Mann–Whitney U ranks; returns None if class missing."""
    labeled = [(yt, yp) for yt, yp in zip(y_true, y_prob) if yt is not None]
    if not labeled:
        return None
    pos = [yp for yt, yp in labeled if yt >= 0.5]
    neg = [yp for yt, yp in labeled if yt < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None

    all_scores = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda x: x[0])
    ranks: List[float] = [0.0] * len(all_scores)

    i = 0
    while i < len(all_scores):
        j = i
        while j + 1 < len(all_scores) and all_scores[j + 1][0] == all_scores[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum_pos = sum(r for (r, (_, lbl)) in zip(ranks, all_scores) if lbl == 1)
    n_pos = len(pos)
    n_neg = len(neg)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def _ks_stat(y_true: List[Optional[float]], y_prob: List[float]) -> Optional[float]:
    """KS statistic between positive & negative score distributions."""
    labeled = [(yt, yp) for yt, yp in zip(y_true, y_prob) if yt is not None]
    if not labeled:
        return None
    pos = sorted([yp for yt, yp in labeled if yt >= 0.5])
    neg = sorted([yp for yt, yp in labeled if yt < 0.5])
    if len(pos) == 0 or len(neg) == 0:
        return None

    i = j = 0
    n_pos, n_neg = len(pos), len(neg)
    ks = 0.0
    values = sorted(set(pos + neg))
    for v in values:
        while i < n_pos and pos[i] <= v:
            i += 1
        while j < n_neg and neg[j] <= v:
            j += 1
        cdf_pos = i / n_pos
        cdf_neg = j / n_neg
        ks = max(ks, abs(cdf_pos - cdf_neg))
    return ks


def _round_or_none(x: Optional[float], ndigits: int = 6) -> Optional[float]:
    return None if x is None else round(float(x), ndigits)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, payload: Dict[str, Optional[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in payload.items():
            w.writerow([k, "" if v is None else v])


def main(preds_path: str, out_dir: str) -> Dict[str, Optional[float]]:
    """Public entrypoint: compute + persist metrics; return metrics dict."""
    preds_p = Path(preds_path)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    y_true, y_prob = _read_predictions(preds_p)

    metrics: Dict[str, Optional[float]] = {
        "n": len(y_prob),
        "accuracy@0.5": _round_or_none(_accuracy(y_true, y_prob, threshold=0.5)),
        "auroc": _round_or_none(_roc_auc(y_true, y_prob)),
        "ks_stat": _round_or_none(_ks_stat(y_true, y_prob)),
    }

    _write_json(out_p / "performance_metrics.json", metrics)
    _write_csv(out_p / "performance_metrics.csv", metrics)

    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute performance metrics and write artifacts.")
    ap.add_argument("--preds", required=True, help="Path to predictions CSV")
    ap.add_argument("--out", default="reports", help="Output directory (default: reports)")
    args = ap.parse_args()
    result = main(args.preds, args.out)
    print(json.dumps(result, indent=2))

