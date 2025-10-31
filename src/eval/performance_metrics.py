# src/eval/performance_metrics.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

EPS = 1e-15


# --------- helpers (pure Python; numpy not required) ---------
def _to_lists(y_true: Sequence, y_score: Sequence) -> Tuple[list[int], list[float]]:
    yt, ys = [], []
    for a, b in zip(y_true, y_score):
        try:
            yt.append(int(a))
            ys.append(float(b))
        except Exception:
            continue
    return yt, ys


def _accuracy_at_threshold(
    y_true: list[int], y_score: list[float], thr: float = 0.5
) -> Optional[float]:
    n = len(y_true)
    if n == 0:
        return None
    correct = sum(((1 if s >= thr else 0) == y) for y, s in zip(y_true, y_score))
    return correct / n


def _auroc(y_true: list[int], y_score: list[float]) -> Optional[float]:
    n = len(y_true)
    if n == 0:
        return None
    pos_idx = [i for i, y in enumerate(y_true) if y == 1]
    neg_idx = [i for i, y in enumerate(y_true) if y == 0]
    n1, n0 = len(pos_idx), len(neg_idx)
    if n1 == 0 or n0 == 0:
        return None
    all_scored = sorted([(s, i) for i, s in enumerate(y_score)], key=lambda x: x[0])
    ranks = {}
    rank = 1
    i = 0
    while i < len(all_scored):
        j = i
        while j + 1 < len(all_scored) and all_scored[j + 1][0] == all_scored[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[all_scored[k][1]] = avg_rank
        rank = j + 2
        i = j + 1
    R1 = sum(ranks[i] for i in pos_idx)
    U1 = R1 - n1 * (n1 + 1) / 2.0
    return U1 / (n1 * n0)


def _ks_stat(y_true: list[int], y_score: list[float]) -> Optional[float]:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n = len(pairs)
    if n == 0:
        return None
    n1 = sum(1 for _, y in pairs if y == 1)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return None
    tp = fp = 0
    maxdiff = 0.0
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n1
        fpr = fp / n0
        d = abs(tpr - fpr)
        if d > maxdiff:
            maxdiff = d
    return maxdiff


def _auprc(y_true: list[int], y_score: list[float]) -> Optional[float]:
    n = len(y_true)
    if n == 0:
        return None
    P = sum(y_true)
    if P == 0:
        return None

    uniq = sorted(set(y_score), reverse=True)
    thresholds = uniq + [min(uniq) - 1] if uniq else [1.0, 0.0]
    prev_recall = 0.0
    area = 0.0
    for t in thresholds:
        tp = fp = 0
        for y, s in zip(y_true, y_score):
            if s >= t:
                if y == 1:
                    tp += 1
                else:
                    fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / P
        area += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return area


def _log_loss(y_true: list[int], y_score: list[float]) -> Optional[float]:
    import math

    n = len(y_true)
    if n == 0:
        return None
    loss = 0.0
    for y, p in zip(y_true, y_score):
        p = max(EPS, min(1 - EPS, p))
        loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return loss / n


def _best_f1(
    y_true: Sequence, y_score: Sequence
) -> Tuple[Optional[float], Optional[float]]:
    yt, ys = _to_lists(y_true, y_score)
    n = len(yt)
    if n == 0:
        return None, None
    candidates = sorted(set([0.0, 1.0] + ys))
    best_f1 = -1.0
    best_t = None
    for t in candidates:
        preds = [(1 if s >= t else 0) for s in ys]
        tp = sum(1 for y, p in zip(yt, preds) if y == 1 and p == 1)
        fp = sum(1 for y, p in zip(yt, preds) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(yt, preds) if y == 1 and p == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        if (f1 > best_f1) or (
            f1 == best_f1 and best_t is not None and abs(t - 0.5) < abs(best_t - 0.5)
        ):
            best_f1 = f1
            best_t = t
    return best_t, best_f1 if best_f1 >= 0 else None


def _round_opt(v: Optional[float], k: int = 6) -> Optional[float]:
    return round(v, k) if (v is not None) else None


# --------- main compute_metrics ---------
def compute_metrics(y_true: Sequence, y_score: Sequence) -> Dict[str, Optional[float]]:
    yt, ys = _to_lists(y_true, y_score)
    n = len(yt)
    P = sum(1 for v in yt if v == 1)
    pos_rate = (P / n) if n else None

    # threshold = 0.5 metrics
    preds = [(1 if s >= 0.5 else 0) for s in ys]
    tp = sum(1 for y, p in zip(yt, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(yt, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(yt, preds) if y == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_at_05 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    # best F1 sweep
    best_t, best_f1 = _best_f1(yt, ys)

    acc = _accuracy_at_threshold(yt, ys, 0.5)
    auroc = _auroc(yt, ys)
    ks = _ks_stat(yt, ys)
    auprc = _auprc(yt, ys)
    ll = _log_loss(yt, ys)

    return {
        "n": n,
        "n_samples": n,
        "positive_rate": round(pos_rate, 6) if pos_rate is not None else None,
        "accuracy@0.5": _round_opt(acc),
        "f1@0.5": round(f1_at_05, 6),
        "best_f1_threshold": round(best_t, 6) if best_t is not None else None,
        "best_f1": round(best_f1, 6) if best_f1 is not None else None,
        "auroc": _round_opt(auroc),
        "ks_stat": _round_opt(ks),
        "auprc": _round_opt(auprc),
        "log_loss": _round_opt(ll),
    }


# --------- artifact writer ---------
def compute_performance_metrics(
    preds_csv: str, out_dir: str = "reports"
) -> Dict[str, Optional[float]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y, p = [], []
    with Path(preds_csv).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldmap = {k.lstrip("\ufeff"): k for k in (r.fieldnames or [])}
        ycol = fieldmap.get("y_true", "y_true")
        pcol = fieldmap.get("y_pred_prob", "y_pred_prob")
        for row in r:
            try:
                y.append(int(row[ycol]))
                p.append(float(row[pcol]))
            except Exception:
                continue

    metrics = compute_metrics(y, p)
    (out / "performance_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (out / "performance_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    return metrics


# --------- CLI entry ---------
def main(
    preds_csv: Optional[str] = None, out_dir: str = "reports"
) -> Dict[str, Optional[float]]:
    if preds_csv is None:
        parser = argparse.ArgumentParser(
            description="Compute performance metrics & write artifacts"
        )
        parser.add_argument(
            "--preds",
            dest="preds_csv",
            required=True,
            help="Path to CSV with y_true,y_pred_prob",
        )
        parser.add_argument(
            "--out_dir", default=out_dir, help="Output directory (default: reports)"
        )
        args = parser.parse_args()
        preds_csv = args.preds_csv
        out_dir = args.out_dir
    return compute_performance_metrics(preds_csv, out_dir)
