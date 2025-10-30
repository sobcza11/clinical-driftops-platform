# src/ops/evidence_digest.py
# Purpose: Produce a cryptographic integrity report of key artifacts.
# Output: reports/evidence_digest.json (SHA256, size, mtime for each file)

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import hashlib
import json
import os
import time

REPORTS = Path("reports")

# Files to include from repo root (outside reports)
ROOT_FILES = [
    Path("policy.yaml"),
    Path("policy_registry.yaml"),
]

# Files to include within reports (relative paths)
REPORT_FILES = [
    "live_validation.json",
    "policy_gate_result.json",
    "performance_metrics.json",
    "performance_metrics.csv",
    "fairness_summary.json",
    "api_fairness_report.md",
    "api_fairness_metrics.csv",
    "shap_top_features.json",
    "regulatory_monitor.json",
    "run_metadata.json",
    "policy_registry_summary.json",
    "index.html",
    "driftops_bundle.zip",
]

def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _file_info(path: Path) -> Dict[str, Any]:
    try:
        stat = path.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "mtime_epoch": int(stat.st_mtime),
            "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
            "sha256": _sha256_file(path),
        }
    except FileNotFoundError:
        return {"exists": False}

def main(out_dir: str = "reports") -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    evidence: Dict[str, Any] = {
        "generator": {"module": "src.ops.evidence_digest", "version": "1.0.0"},
        "root_files": {},
        "report_files": {},
    }

    # Root-level YAML/policy files
    for p in ROOT_FILES:
        evidence["root_files"][str(p)] = _file_info(p)

    # Report artifacts
    for name in REPORT_FILES:
        p = out / name
        evidence["report_files"][name] = _file_info(p)

    target = out / "evidence_digest.json"
    target.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return str(target)

if __name__ == "__main__":
    print(main())
