# src/ops/mlflow_utils.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional

# Soft dependency: no-ops if mlflow isn't installed
try:
    import mlflow
except Exception:  # noqa: BLE001
    mlflow = None  # type: ignore[misc]


def _active() -> bool:
    return mlflow is not None


def start_run(context: str, nested: bool = True) -> Optional[str]:
    """
    Start (or reuse) an MLflow run. Safe no-op if mlflow missing.
    Returns run_id or None.
    """
    if not _active():
        return None

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Clinical DriftOps")
    mlflow.set_experiment(exp_name)

    # If there's already an active run, optionally nest
    if mlflow.active_run() is None:
        run = mlflow.start_run(run_name=f"{context} | {os.getenv('GITHUB_SHA','')[:7]}", nested=False)
    else:
        run = mlflow.start_run(run_name=context, nested=nested)

    # Enrich with useful CI tags
    tags = {
        "context": context,
        "commit": os.getenv("GITHUB_SHA", "")[:40],
        "branch": os.getenv("GITHUB_REF_NAME", ""),
        "repo": os.getenv("GITHUB_REPOSITORY", ""),
        "run_id": os.getenv("GITHUB_RUN_ID", ""),
        "actor": os.getenv("GITHUB_ACTOR", ""),
        "ci": "github_actions" if os.getenv("GITHUB_ACTIONS") else "local",
    }
    mlflow.set_tags({k: v for k, v in tags.items() if v})

    return run.info.run_id if run else None


def log_params(params: Dict) -> None:
    if not _active():
        return
    mlflow.log_params(params)


def log_metrics(metrics: Dict, step: Optional[int] = None) -> None:
    if not _active():
        return
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str | Path, artifact_path: Optional[str] = None) -> None:
    if not _active():
        return
    p = Path(path)
    if p.exists():
        mlflow.log_artifact(str(p), artifact_path=artifact_path)


def log_artifacts(path: str | Path, artifact_path: Optional[str] = None) -> None:
    if not _active():
        return
    p = Path(path)
    if p.exists():
        mlflow.log_artifacts(str(p), artifact_path=artifact_path)
