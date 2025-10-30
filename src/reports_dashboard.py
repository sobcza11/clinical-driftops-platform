# src/reports_dashboard.py
# Purpose: Build a static HTML dashboard from artifacts in reports/
# Required by tests:
#   - Title: "Clinical DriftOps — Reports Dashboard"
#   - Contains the section "Policy Gate"
#
# Sections:
#   - Run Status badge (from live_validation.json)
#   - Policy Gate (pretty-printed JSON)
#   - Policy Thresholds vs Actuals (computed from policy + performance)
#   - Performance (n, accuracy@0.5, auroc, ks_stat)
#   - Top Features (SHAP) if shap_top_features.json exists
#   - Fairness Slices (from fairness_summary.json)

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

def _policy_table(gate: Dict[str, Any], perf: Dict[str, Any]) -> str:
    # Extract thresholds from gate["policy"]
    policy = gate.get("policy", {}) if isinstance(gate, dict) else {}
    min_auroc = policy.get("min_auroc")
    min_ks = policy.get("min_ks")

    # Extract actuals
    auroc = perf.get("auroc")
    ks = perf.get("ks_stat")

    rows: List[str] = []
    def row(name: str, actual: Any, threshold: Any, higher_is_better: bool = True) -> None:
        try:
            a = float(actual) if actual is not None else None
            t = float(threshold) if threshold is not None else None
        except Exception:
            a, t = None, None
        ok = None
        if a is not None and t is not None:
            ok = (a >= t) if higher_is_better else (a <= t)
        status = "PASS" if ok else ("—" if ok is None else "FAIL")
        color = "#22c55e" if status == "PASS" else ("#9ca3af" if status == "—" else "#ef4444")
        rows.append(
            f"<tr><td>{name}</td><td>{'' if a is None else a}</td>"
            f"<td>{'' if t is None else t}</td>"
            f"<td><b style='color:{color}'>{status}</b></td></tr>"
        )

    row("AUROC", auroc, min_auroc, higher_is_better=True)
    row("KS Statistic", ks, min_ks, higher_is_better=True)

    table = f"""
      <section>
        <h2>Policy Thresholds vs Actuals</h2>
        <table border="1" cellspacing="0" cellpadding="6">
          <thead><tr><th>Metric</th><th>Actual</th><th>Threshold</th><th>Status</th></tr></thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </section>
    """
    return table

def _shap_section(shap: Dict[str, Any]) -> str:
    feats: List[Dict[str, Any]] = shap.get("features", [])
    if not feats:
        return ""
    rows = "".join(
        f"<tr><td>{i+1}</td><td>{f.get('name','')}</td><td>{f.get('mean_abs_impact','')}</td></tr>"
        for i, f in enumerate(feats)
    )
    return f"""
      <section>
        <h2>Top Features (SHAP)</h2>
        <table border="1" cellspacing="0" cellpadding="6">
          <thead><tr><th>#</th><th>Feature</th><th>Mean |impact|</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </section>
    """

def _fairness_section(fair: Dict[str, Any]) -> str:
    # Expected shape (placeholder or real):
    # {
    #   "slices": ["overall", "age<40", "age>=40", ...],
    #   "metrics": {
    #       "overall": {"demographic_parity_ratio": 1.0, "equalized_odds": 0.98, ...},
    #       "age<40": {...}
    #   }
    # }
    slices = fair.get("slices", [])
    metrics_by_slice: Dict[str, Dict[str, Any]] = fair.get("metrics", {})

    # Collect all metric names across slices for a stable header
    metric_names: List[str] = []
    for s in slices:
        for m in (metrics_by_slice.get(s, {}) or {}).keys():
            if m not in metric_names:
                metric_names.append(m)

    if not slices or not metric_names:
        return ""

    # Build header
    thead = "<thead><tr><th>Slice</th>" + "".join(f"<th>{m}</th>" for m in metric_names) + "</tr></thead>"

    # Build rows
    body_rows: List[str] = []
    for s in slices:
        vals = metrics_by_slice.get(s, {}) or {}
        tds = "".join(f"<td>{'' if vals.get(m) is None else vals.get(m)}</td>" for m in metric_names)
        body_rows.append(f"<tr><td>{s}</td>{tds}</tr>")

    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"

    return f"""
      <section>
        <h2>Fairness Slices</h2>
        <table border="1" cellspacing="0" cellpadding="6">
          {thead}
          {tbody}
        </table>
      </section>
    """

def build() -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)

    live = _read_json(REPORTS / "live_validation.json")
    gate = _read_json(REPORTS / "policy_gate_result.json")
    perf = _read_json(REPORTS / "performance_metrics.json")
    shap = _read_json(REPORTS / "shap_top_features.json")
    fair = _read_json(REPORTS / "fairness_summary.json")

    status_badge = _badge(live.get("status", "FAIL"))

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
    pre {{ background:#0b1221; color:#e5e7eb; padding:12px; border-radius:8px; overflow:auto; }}
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

  {_policy_table(gate, perf)}

  <section>
    <h2>Performance</h2>
    <table class="kv" border="0" cellspacing="0" cellpadding="0">
      <tr><td>n</td><td>{perf.get('n','')}</td></tr>
      <tr><td>accuracy@0.5</td><td>{perf.get('accuracy@0.5','')}</td></tr>
      <tr><td>auroc</td><td>{perf.get('auroc','')}</td></tr>
      <tr><td>ks_stat</td><td>{perf.get('ks_stat','')}</td></tr>
    </table>
  </section>

  {_shap_section(shap)}
  {_fairness_section(fair)}

</body>
</html>
"""
    target = REPORTS / "index.html"
    target.write_text(html, encoding="utf-8")
    return str(target)

if __name__ == "__main__":
    print(build())


