# src/ops/policy_registry_view.py
# Purpose: Summarize policy.yaml and policy_registry.yaml into a normalized JSON
# Output: reports/policy_registry_summary.json
# Behavior:
#   - Writes the summary even if YAML files are missing.
#   - Works even if PyYAML is not installed (falls back to "present: false" + empty data).

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json

REPORTS = Path("reports")


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML if possible; otherwise return {}. Missing file -> {}."""
    if not path.exists():
        return {}
    try:
        import yaml  # optional dependency
    except Exception:
        # PyYAML not installed; return {} (we still emit a summary file)
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def main(out_dir: str = "reports") -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    policy_path = Path("policy.yaml")
    registry_path = Path("policy_registry.yaml")

    policy = _safe_load_yaml(policy_path)
    registry = _safe_load_yaml(registry_path)

    thresholds = (policy.get("thresholds") or {}) if isinstance(policy, dict) else {}
    settings = (policy.get("settings") or {}) if isinstance(policy, dict) else {}

    policies: List[Dict[str, Any]] = []
    if isinstance(registry, dict):
        for item in registry.get("policies") or []:
            if isinstance(item, dict):
                policies.append(
                    {
                        "id": item.get("id"),
                        "description": item.get("description"),
                        "thresholds": item.get("thresholds"),
                        "applies_to": item.get("applies_to"),
                    }
                )

    summary = {
        "policy_yaml_present": policy_path.exists(),
        "policy_registry_yaml_present": registry_path.exists(),
        "active_policy": {
            "thresholds": {
                "min_auroc": thresholds.get("min_auroc"),
                "min_ks": thresholds.get("min_ks"),
            },
            "settings": {
                "allow_missing_labels": settings.get("allow_missing_labels"),
            },
        },
        "registry": {
            "policies": policies,
            "note": "Add entries to policy_registry.yaml to map contexts to policies.",
        },
        "generator": {"module": "src.ops.policy_registry_view", "version": "1.0.0"},
    }

    target = out / "policy_registry_summary.json"
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(target)


if __name__ == "__main__":
    print(main())
