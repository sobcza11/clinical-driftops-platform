# src/ops/drift_history.py
# Purpose: Maintain a rolling history of key validation metrics across runs.
# Output: reports/drift_history.json (list of records)

from __future__ import annotations
from pathlib import Path
import json
import os
from datetime import datetime, timezone

REPORTS = Path("reports")
HISTORY_FILE = REPORTS / "drift_history.json"
MAX_RECORDS = 200  # rolling window


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [] if path == HISTORY_FILE else {}


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(out_dir: str = "reports") -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    live = _read_json(out / "live_validation.json") or {}
    perf = _read_json(out / "performance_metrics.json") or {}
    gate = _read_json(out / "policy_gate_result.json") or {}

    record = {
        "ts": _iso_now(),
        "status": (live.get("status") or "").upper(),
        "auroc": perf.get("auroc"),
        "ks_stat": perf.get("ks_stat"),
        "min_auroc": (gate.get("policy") or {}).get("min_auroc"),
        "min_ks": (gate.get("policy") or {}).get("min_ks"),
        "reasons": gate.get("reasons", []),
        "ci": {
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_number": os.environ.get("GITHUB_RUN_NUMBER"),
            "repo": os.environ.get("GITHUB_REPOSITORY"),
            "sha": os.environ.get("GITHUB_SHA"),
            "ref": os.environ.get("GITHUB_REF"),
        },
    }

    history = _read_json(HISTORY_FILE)
    if not isinstance(history, list):
        history = []

    history.append(record)
    if len(history) > MAX_RECORDS:
        history = history[-MAX_RECORDS:]

    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return str(HISTORY_FILE)


if __name__ == "__main__":
    print(main())
