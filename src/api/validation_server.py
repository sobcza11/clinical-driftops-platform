# clinical-driftops-platform/src/api/validation_server.py
from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from typing import Optional, Dict, Any
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timezone

# Reuse your existing logic
from src.eval.performance_metrics import (
    compute_from_predictions_csv as _perf_from_preds,
)  # adjust if different
from src.eval.fairness_audit import audit_fairness as _audit_fairness
from src.ops.policy_gate import main as _run_policy_gate

API_TOKEN = None  # read from env in __main__ when you run the server

app = FastAPI(title="Clinical DriftOps Validation API")


def _ts() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@app.post("/validate")
async def validate(
    file: UploadFile = File(...), authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    # simple token check
    if API_TOKEN and (authorization or "").replace("Bearer ", "") != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Load CSV (can be a predictions file with y_true,y_score OR a prepared dataset)
    df = pd.read_csv(file.file)

    # If it looks like predictions -> compute performance; else skip gracefully
    perf = {}
    if {"y_true", "y_score"}.issubset(set(df.columns)):
        tmp_path = Path("reports/api_predictions.csv")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        df[["y_true", "y_score"]].to_csv(tmp_path, index=False)
        perf = (
            json.loads(Path("reports/performance_metrics.json").read_text())
            if _perf_from_preds(str(tmp_path)) is None
            else {}
        )

    # Fairness audit on provided dataset if possible (fallback to group autodetect)
    fair_csv = "reports/api_fairness_metrics.csv"
    fair_md = "reports/api_fairness_report.md"
    try:
        _audit_fairness(
            "data/data_prepared_current.csv", None, fair_csv, fair_md
        )  # use current prepared data
    except Exception:
        pass

    # Run gate (writes reports/policy_gate_result.json)
    gate_code = _run_policy_gate()
    gate = {}
    try:
        gate = json.loads(
            Path("reports/policy_gate_result.json").read_text(encoding="utf-8")
        )
    except Exception:
        pass

    result = {
        "timestamp_utc": _ts(),
        "status": "PASS" if gate_code == 0 else "FAIL",
        "performance": perf,
        "gate": gate,
    }
    Path("reports/live_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    import os

    API_TOKEN = os.getenv("DRIFTOPS_API_TOKEN")
    import uvicorn

    uvicorn.run(
        "src.api.validation_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )
