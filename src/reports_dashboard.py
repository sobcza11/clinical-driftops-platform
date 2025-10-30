from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

REPORTS = Path("reports")

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _badge(status: str) -> str:
    s = (status or "").upper()
    color = "#22c55e" if s == "PASS" else "#ef4444"
    text = "PASS" if s == "PASS" else "FAIL"
    return f'<span style="display:inline-block;padding:4px 10px;border-radius:9999px;background:{color};color:#fff;font-weight:600;">{text}</span>'

def build() -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)

    live = _read_json(REPORTS / "live_validation.json")
    gate = _read_json(REPORTS / "policy_gate_result.json")
    perf = _read_json(REPORTS / "performance_metrics.json")
    shap = _read_json(REPORTS / "shap_top_features.json")

    status_badge = _badge(live.get("status", "FAIL"))

    shap_rows = ""
    feats: List[Dict[str, Any]] = shap.get("features", [])
    if feats:
        shap_rows = "".join(
            f"<tr><td>{i+1}</td><td>{f.get('name','')}</td><td>{f.get('mean_abs_impact','')}</td></tr>"
            for i, f in enumerate(feats)
        )
        shap_section = f"""
        <section>
          <h2>Top Features (SHAP)</h2>
          <table border="1" cellspacing="0" cellpadding="6">
            <thead><tr><th>#</th><th>Feature</th><th>Mean |impact|</th></tr></thead>
            <tbody>{shap_rows}</tbody>
          </table>
        </section>
        """
    else:
        shap_section = ""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Clinical DriftOps — Reports Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 24px; }}
    h1,h2 {{ margin: 0 0 12px; }}
    section {{ margin: 20px 0; }}
    .kv td {{ padding: 4px 8px; }}
  </style>
</head>
<body>
  <header>
    <h1>Clinical DriftOps — Reports Dashboard</h1>
    <div>Run Status: {status_badge}</div>
  </header>

  <section>
    <h2>Policy Gate</h2>
    <pre>{json.dumps(gate, indent=2)}</pre>
  </section>

  <section>
    <h2>Performance</h2>
    <table class="kv" border="0" cellspacing="0" cellpadding="0">
      <tr><td>n</td><td>{perf.get('n','')}</td></tr>
      <tr><td>accuracy@0.5</td><td>{perf.get('accuracy@0.5','')}</td></tr>
      <tr><td>auroc</td><td>{perf.get('auroc','')}</td></tr>
      <tr><td>ks_stat</td><td>{perf.get('ks_stat','')}</td></tr>
    </table>
  </section>

  {shap_section}

</body>
</html>
"""
    target = REPORTS / "index.html"
    target.write_text(html, encoding="utf-8")
    return str(target)

if __name__ == "__main__":
    print(build())


