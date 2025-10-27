# clinical-driftops-platform/src/reports_dashboard.py
from __future__ import annotations
import json, base64
from pathlib import Path
from datetime import datetime

import pandas as pd

def embed_image(path: Path) -> str:
    """Return <img> tag with base64 inline PNG if file exists."""
    if not path.exists():
        return f"<p style='color:gray;'>Missing: {path.name}</p>"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"<img src='data:image/png;base64,{data}' alt='{path.name}' style='max-width:640px;border:1px solid #ccc;border-radius:8px;'>"


def df_to_html(path: Path, title: str) -> str:
    if not path.exists():
        return f"<h3>{title}</h3><p style='color:gray;'>Missing: {path.name}</p>"
    try:
        df = pd.read_csv(path)
    except Exception:
        return f"<h3>{title}</h3><p style='color:red;'>Error reading {path.name}</p>"
    return f"<h3>{title}</h3>{df.to_html(index=False, escape=False)}"


def main() -> None:
    rpt = Path("reports")
    out = rpt / "index.html"
    rpt.mkdir(exist_ok=True)

    # Load gate JSON
    gate_path = rpt / "policy_gate_result.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}

    status = gate.get("status", "UNKNOWN")
    ts = gate.get("timestamp_utc", "")
    observed = gate.get("observed", {})
    checks = pd.DataFrame(gate.get("checks", []))

    # HTML build
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Clinical DriftOps Dashboard</title>")
    html.append("""
    <style>
      body{font-family:Segoe UI,Roboto,sans-serif;margin:40px;color:#222;}
      h1,h2,h3{color:#004b87;}
      table{border-collapse:collapse;margin-bottom:24px;}
      th,td{border:1px solid #ddd;padding:6px 10px;}
      th{background:#f2f2f2;}
      .pass{color:green;font-weight:600;}
      .fail{color:red;font-weight:600;}
      .meta{font-size:0.9em;color:#555;margin-bottom:20px;}
    </style></head><body>
    """)
    html.append("<h1>🏥 Clinical DriftOps Dashboard</h1>")
    html.append(f"<div class='meta'>Generated {datetime.utcnow().isoformat(timespec='seconds')} UTC</div>")
    html.append(f"<h2>Gate Status: <span class='{'pass' if status=='PASS' else 'fail'}'>{status}</span></h2>")
    html.append(f"<p><strong>Timestamp UTC:</strong> {ts}</p>")

    # Observed summary
    if observed:
        html.append("<h3>Observed Metrics</h3><ul>")
        for k,v in observed.items():
            html.append(f"<li>{k}: <b>{v}</b></li>")
        html.append("</ul>")

    # Checks table
    if not checks.empty:
        html.append("<h3>Policy Checks</h3>")
        html.append(checks.to_html(index=False, escape=False))

    # Drift table
    html.append(df_to_html(rpt / "drift_metrics.csv", "Drift Metrics"))
    # Fairness
    html.append(df_to_html(rpt / "fairness_metrics.csv", "Fairness Metrics"))
    # Embed SHAP
    html.append("<h3>SHAP Top Features</h3>")
    html.append(embed_image(rpt / "shap_top_features.png"))

    # Trustworthy AI audit (markdown preview)
    audit_md = rpt / "trustworthy_ai_audit_v1.md"
    if audit_md.exists():
        html.append("<h3>Trustworthy AI Audit</h3><pre style='background:#fafafa;border:1px solid #ccc;padding:12px;border-radius:6px;'>")
        html.append(audit_md.read_text(encoding="utf-8"))
        html.append("</pre>")

    html.append("</body></html>")
    out.write_text("\n".join(html), encoding="utf-8")
    print(f"[Dashboard] Complete → {out.resolve()}")

if __name__ == "__main__":
    main()
