# src/ops/run_metadata.py
# Purpose: Emit helpful run links for operators/auditors.
# Writes: reports/run_metadata.json

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List

REPORTS = Path("reports")

def _maybe_mlruns() -> List[Dict[str, Any]]:
    base = Path("mlruns")
    results: List[Dict[str, Any]] = []
    if not base.exists():
        return results

    # Very light scan for meta.yaml files
    for meta in base.rglob("meta.yaml"):
        try:
            rel = meta.parent.relative_to(Path("."))
            # Keep path & a short name
            results.append({
                "name": rel.name,
                "path": str(rel).replace("\\", "/"),
            })
        except Exception:
            continue
    # Limit to a handful for readability
    return results[:5]

def main(out_dir: str = "reports") -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)

    server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repo   = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    actions_run_url = f"{server}/{repo}/actions/runs/{run_id}" if repo and run_id else ""

    # Pages URL (static for user repo)
    user_repo = repo.split("/") if repo else []
    pages_url = ""
    if len(user_repo) == 2:
        owner, name = user_repo
        pages_url = f"https://{owner}.github.io/{name}/"

    payload = {
        "ci": {
            "actions_run_url": actions_run_url,
            "pages_url": pages_url,
        },
        "mlflow": {
            "runs": _maybe_mlruns(),
            "note": "If using MLflow tracking, link a public UI; paths listed when mlruns/** exists."
        }
    }

    target = Path(out_dir) / "run_metadata.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(target)

if __name__ == "__main__":
    print(main())
