from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _load_policy(root: Path) -> Dict[str, Any]:
    policy_defaults = {
        "min_auroc": 0.70,
        "min_ks": 0.10,
        "allow_missing_labels": True,
    }
    yaml_file = root / "policy.yaml"
    if yaml_file.exists():
        try:
            import yaml
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
            return {
                "min_auroc": float(data.get("thresholds", {}).get("min_auroc", policy_defaults["min_auroc"])),
                "min_ks": float(data.get("thresholds", {}).get("min_ks", policy_defaults["min_ks"])),
                "allow_missing_labels": bool(data.get("settings", {}).get("allow_missing_labels", policy_defaults["allow_missing_labels"])),
            }
        except Exception:
            pass
    return policy_defaults

def main(out_dir: str = "reports", performance: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out = Path(out_dir)
    root = Path(".").resolve()
    out.mkdir(parents=True, exist_ok=True)

    perf = performance or _read_json(out / "performance_metrics.json")

    policy = _load_policy(root)
    reasons: list[str] = []

    auroc = perf.get("auroc", None)
    ks    = perf.get("ks_stat", None)

    status = "PASS"
    if auroc is None or ks is None:
        if not policy.get("allow_missing_labels", True):
            status = "FAIL"; reasons.append("Missing labels prevented AUROC/KS; policy disallows.")
    else:
        if auroc < policy["min_auroc"]:
            status = "FAIL"; reasons.append(f"AUROC {auroc} < min_auroc {policy['min_auroc']}")
        if ks < policy["min_ks"]:
            status = "FAIL"; reasons.append(f"KS {ks} < min_ks {policy['min_ks']}")

    result = {
        "status": status,  # PASS/FAIL
        "reasons": reasons,
        "policy": {
            "min_auroc": policy["min_auroc"],
            "min_ks": policy["min_ks"],
            "allow_missing_labels": policy["allow_missing_labels"],
        },
    }

    (out / "policy_gate_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result

if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

