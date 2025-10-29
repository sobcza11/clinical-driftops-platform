"""
Clinical DriftOps — Unified Dashboard Builder
Generates reports/index.html aggregating metrics, drift, fairness, and Phase VI operational insights.
"""

from __future__ import annotations
import os, json
from datetime import datetime, timezone
from pathlib import Path

# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def _read_json(path: str) -> dict | None:
    """Safely read JSON if it exists."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_head(path: str, n: int = 5) -> list[str]:
    """Return first n lines of a CSV file for preview."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()[:n]
        return lines
    except Exception:
        return []


# -------------------------------------------------------------------
# Main build
# -------------------------------------------------------------------

def build_dashboard() -> str:
    os.makedirs("reports", exist_ok=True)
    html: list[str] = []

    html.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html.append("<title>Clinical DriftOps Dashboard</title>")
    html.append(
        "<style>"
        "body{font-family:Segoe UI,Roboto,Arial,sans-serif;margin:2em;line-height:1.5;background:#fafafa;color:#222;}"
        "h1,h2{color:#0b3d91;}"
        ".card{background:#fff;border-radius:12px;padding:1em;margin-bottom:1.5em;"
        "box-shadow:0 2px 5px rgba(0,0,0,0.08);}"
        "ul{margin:0.5em 0 0.5em 1.2em;}"
        ".meta{font-size:0.85em;color:#555;margin-top:1em;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1em;}"
        ".tile{background:#f9f9f9;border-radius:8px;padding:0.7em;}"
        "</style>"
    )
    html.append("</head><body>")
    html.append("<h1>🩺 Clinical DriftOps Dashboard</h1>")
    html.append("<p>Automatically generated via GitHub Actions CI/CD.</p>")

    # -------------------------------------------------------------------
    # Phase V: Performance metrics
    # -------------------------------------------------------------------
    perf = _read_json("reports/performance_metrics.json")
    if perf:
        html.append("<section class='card'><h2>Performance Metrics</h2><ul>")
        for k, v in perf.items():
            html.append(f"<li>{k}: {v}</li>")
        html.append("</ul></section>")

    # -------------------------------------------------------------------
    # Drift metrics
    # -------------------------------------------------------------------
    drift_head = _read_csv_head("reports/drift_metrics.csv")
    if drift_head:
        html.append("<section class='card'><h2>Data Drift</h2><pre>")
        html.append("\n".join(drift_head))
        html.append("</pre></section>")

    # -------------------------------------------------------------------
    # Fairness metrics
    # -------------------------------------------------------------------
    fairness_head = _read_csv_head("reports/fairness_metrics.csv")
    if fairness_head:
        html.append("<section class='card'><h2>Fairness Metrics</h2><pre>")
        html.append("\n".join(fairness_head))
        html.append("</pre></section>")

    # -------------------------------------------------------------------
    # SHAP explainability
    # -------------------------------------------------------------------
    shap_path = Path("reports/shap_top_features.png")
    if shap_path.exists():
        html.append("<section class='card'><h2>SHAP Explainability</h2>")
        html.append(f"<img src='{shap_path.name}' style='max-width:100%;border-radius:8px;'>")
        html.append("</section>")

    # -------------------------------------------------------------------
    # Phase VI — Live Validation
    # -------------------------------------------------------------------
    live = _read_json("reports/live_validation.json")
    if live:
        html.append("<section class='card'>")
        html.append("<h2>Live Validation</h2>")
        html.append(f"<div>Status: <b>{live.get('status','?')}</b> • {live.get('timestamp_utc','')}</div>")
        perf_live = live.get("performance") or {}
        if perf_live:
            html.append("<ul>")
            for k in ("auroc","auprc","log_loss","f1@0.5","f1@bestF1","best_f1_threshold"):
                if k in perf_live:
                    html.append(f"<li>{k}: {perf_live[k]}</li>")
            html.append("</ul>")
        html.append("</section>")

    # -------------------------------------------------------------------
    # Phase VI — Compliance Monitor
    # -------------------------------------------------------------------
    cmpx = _read_json("reports/compliance_updates.json")
    if cmpx:
        html.append("<section class='card'>")
        html.append("<h2>Compliance Monitor</h2>")
        html.append(f"<div>Last check: {cmpx.get('timestamp_utc','')}</div>")
        updates = cmpx.get("updates") or {}
        html.append("<div class='grid'>")
        for domain, items in updates.items():
            html.append("<div class='tile'>")
            html.append(f"<h3>{domain}</h3>")
            if items:
                html.append("<ul>")
                for it in items[:3]:
                    title = it.get("title","Update")
                    src = it.get("source","")
                    date = it.get("date","")
                    html.append(f"<li>{title} — <small>{date}</small><br><code>{src}</code></li>")
                html.append("</ul>")
            else:
                html.append("<p><i>No recent items.</i></p>")
            html.append("</div>")
        html.append("</div></section>")

    # -------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------
    html.append(
        f"<div class='meta'>Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC</div>"
    )
    html.append("</body></html>")

    out_path = Path("reports/index.html")
    out_path.write_text("\n".join(html), encoding="utf-8")
    print(f"✅ Dashboard built → {out_path}")
    return str(out_path)


if __name__ == "__main__":
    build_dashboard()
