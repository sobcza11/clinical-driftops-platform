from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

REPORTS = Path("reports")

def _read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)

    gate = _read_json(REPORTS / "policy_gate_result.json", {})
    perf = _read_json(REPORTS / "performance_metrics.json", {})
    fair = _read_json(REPORTS / "fairness_summary.json", {})
    shap = _read_json(REPORTS / "shap_top_features.json", {})
    live = _read_json(REPORTS / "live_validation.json", {})

    # IMPORTANT: exact em dash in both title and h1
    title = "Clinical DriftOps — Reports Dashboard"

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>{title}</title>
</head><body>
<h1>{title}</h1>
<p>Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}</p>

<h2>Policy Gate</h2>
<pre>{json.dumps(gate or {}, indent=2)}</pre>

<h2>Performance</h2>
<pre>{json.dumps(perf or {}, indent=2)}</pre>

<h2>Fairness Summary</h2>
<pre>{json.dumps(fair or {}, indent=2)}</pre>

<h2>Explainability (SHAP)</h2>
<pre>{json.dumps(shap or {}, indent=2)}</pre>

<h2>Live Validation</h2>
<pre>{json.dumps(live or {}, indent=2)}</pre>
</body></html>
"""
    (REPORTS / "index.html").write_text(html, encoding="utf-8")
    return 0

if __name__ == "__main__":
    import sys
    raise SystemExit(main())


