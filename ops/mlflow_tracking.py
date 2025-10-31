# Goal: Start tracking metrics/artifacts locally so every CI run is versioned
# src/ops/mlflow_tracking.py
import os
import mlflow
from pathlib import Path


def get_tracking_uri():
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        uri = f"file://{Path.cwd() / 'mlruns'}"
    return uri


def start_run(run_name: str, tags: dict = None):
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment("Clinical-DriftOps")
    return mlflow.start_run(run_name=run_name, tags=tags or {})


def log_params(params: dict):
    [mlflow.log_param(k, v) for k, v in params.items()]


def log_metrics(metrics: dict):
    [mlflow.log_metric(k, float(v)) for k, v in metrics.items()]


def log_artifact(path: str):
    mlflow.log_artifact(path)
