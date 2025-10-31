# src/reports_dashboard.py
# Purpose: Build a static HTML dashboard from artifacts in reports/.
# Note: Tests only require the title and a "Policy Gate" section, but we render more if present.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPORTS = Path("reports")


# ------------------------------- helpers -------------------------------------


def _read_json(path: Path) -> Dict[str, Any] | List[Dict[str, Any]] | Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # Return {} for objects and [] for lists if clearly expected, else {}
        if path.name.endswith(".json"):
            return {}
        return {}


def _exists(p: Path) -> bool:
    return p.exists() and p.is_file()


def _badge(status: str) -> str:
    s = (status or "").upper()
    color = "#22c55e" if s == "PASS" else "#ef4444"
    text = "PASS" if s == "PASS" else "FAIL"
    return (
        '<span style="display:inline-block;padding:4px 10px;border-radius:9999px;'
        f'background:{color};color:#fff;font-weight:600;">{text}</span>'
    )


def _policy_table(gate: Dict[str, Any], perf: Dict[str, Any]) -> str:
    policy = gate.get("policy", {}) if isinstance(gate, dict) else {}
    min_auroc = policy.get("min_auroc")
    min_ks = policy.get("min_ks")
    auroc = perf.get("auroc")
    ks = perf.get("ks_stat")

    rows: List[str] = []

    def row(name: str, actual, threshold, higher_is_better=True):
        try:
            a = float(actual) if actual is not None else None
            t = float(threshold) if threshold is not None else None
        except Exception:
            a, t = None, None

        ok = None
        if a is not None and t is not None:
            ok = (a >= t) if higher_is_better else (a <= t)

        status = "PASS" if ok else ("—" if ok is None else "FAIL")
        color = (
            "#22c55e"
            if status == "PASS"
            else ("#9ca3af" if status == "—" else "#ef4444")
        )

        rows.append(
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{'' if a is None else a}</td>"
            f"<td>{'' if t is None else t}</td>"
            f"<td><b style='color:{color}'>{status}</b></td>"
            "</tr>"
        )

    row("AUROC", auroc, min_auroc, True)
    row("KS Statistic", ks, min_ks, True)

    return f"""
      <section>
        <h2>Policy Thresholds vs Actuals</h2>
        <table border="1" cellspacing="0" cellpadding="6">
          <thead><tr><th>Metric</th><th>Actual</th><th>Threshold</th><th>Status</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </section>
    """


def _shap_section(shap: Dict[str, Any]) -> str:
    feats: List[Dict[str, Any]] = (
        shap.get("features", []) or shap.get("top_features", []) or []
    )
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
    # Accept either {"slices":[...], "metrics":{slice:{metric:value}}}
    # or {"overall": {...}} (tests write overall DPR but don't require rendering)
    slices = fair.get("slices", [])
    metrics_by_slice: Dict[str, Any] = fair.get("metrics", {})

    metric_names: List[str] = []
    for s in slices:
        for m in (metrics_by_slice.get(s, {}) or {}).keys():
            if m not in metric_names:
                metric_names.append(m)

    if not slices or not metric_names:
        return ""

    thead = (
        "<thead><tr><th>Slice</th>"
        + "".join(f"<th>{m}</th>" for m in metric_names)
        + "</tr></thead>"
    )
    body_rows = []
    for s in slices:
        vals = metrics_by_slice.get(s, {}) or {}
        tds = "".join(
            f"<td>{'' if vals.get(m) is None else vals.get(m)}</td>"
            for m in metric_names
        )
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


def _regulatory_section(reg: Dict[str, Any]) -> str:
    rm = reg.get("regulatory_monitor", {}) if isinstance(reg, dict) else {}
    if not rm:
        return ""
    rows = []

    def r(k, v):
        rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")

    r("Policy Gate", rm.get("policy_gate", ""))
    r("Risk Level", rm.get("risk_level", ""))
    r("Explainability Present", rm.get("explainability_present", ""))
    r("Fairness Present", rm.get("fairness_present", ""))
    r("Performance Present", rm.get("performance_present", ""))
    r("Audit Trail Present", rm.get("audit_trail_present", ""))
    r("HIPAA PHI in Artifacts", rm.get("hipaa_phi_in_artifacts", ""))

    notes = rm.get("notes", [])
    notes_html = (
        "<ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>" if notes else ""
    )
    return f"""
      <section>
        <h2>Regulatory Monitor</h2>
        <table border="1" cellspacing="0" cellpadding="6">
          <tbody>{''.join(rows)}</tbody>
        </table>
        {notes_html}
      </section>
    """


def _runmeta_section(meta: Dict[str, Any]) -> str:
    ci = meta.get("ci", {}) if isinstance(meta, dict) else {}
    ml = meta.get("mlflow", {}) if isinstance(meta, dict) else {}
    actions = ci.get("actions_run_url", "")
    pages = ci.get("pages_url", "")
    runs = ml.get("runs", []) or []
    rows = []
    if actions:
        rows.append(
            f"<tr><td>Actions Run</td><td><a href='{actions}' target='_blank' rel='noopener'>{actions}</a></td></tr>"
        )
    if pages:
        rows.append(
            f"<tr><td>Dashboard</td><td><a href='{pages}' target='_blank' rel='noopener'>{pages}</a></td></tr>"
        )
    if runs:
        for r in runs:
            rows.append(
                f"<tr><td>MLflow Run</td><td><code>{r.get('path','')}</code></td></tr>"
            )
    if not rows:
        return ""
    return f"""
      <section>
        <h2>Run Metadata</h2>
        <table border="1" cellspacing="0" cellpadding="6">
          <tbody>{''.join(rows)}</tbody>
        </table>
      </section>
    """


def _policy_registry_section(summary: Dict[str, Any]) -> str:
    if not summary:
        return ""
    active = summary.get("active_policy", {})
    thresholds = active.get("thresholds", {})
    settings = active.get("settings", {})
    registry = summary.get("registry", {}) or {}
    policies = registry.get("policies", []) or []

    rows_a = [
        f"<tr><td>min_auroc</td><td>{thresholds.get('min_auroc','')}</td></tr>",
        f"<tr><td>min_ks</td><td>{thresholds.get('min_ks','')}</td></tr>",
        f"<tr><td>allow_missing_labels</td><td>{settings.get('allow_missing_labels','')}</td></tr>",
    ]
    active_html = f"""
      <h3>Active Policy</h3>
      <table border="1" cellspacing="0" cellpadding="6">
        <tbody>{''.join(rows_a)}</tbody>
      </table>
    """

    reg_html = ""
    if policies:
        head = "<thead><tr><th>Policy ID</th><th>Description</th><th>Applies To</th><th>Thresholds</th></tr></thead>"
        body_rows = []
        for p in policies:
            thr = p.get("thresholds") or {}
            thr_str = ", ".join(f"{k}={v}" for k, v in thr.items())
            body_rows.append(
                f"<tr><td>{p.get('id','')}</td><td>{p.get('description','')}</td>"
                f"<td>{', '.join(p.get('applies_to') or [])}</td>"
                f"<td>{thr_str}</td></tr>"
            )
        reg_html = f"""
          <h3>Registry</h3>
          <table border="1" cellspacing="0" cellpadding="6">
            {head}
            <tbody>{''.join(body_rows)}</tbody>
          </table>
        """

    return f"""
      <section>
        <h2>Policy Registry</h2>
        {active_html}
        {reg_html}
      </section>
    """


def _bundle_link() -> str:
    bundle = REPORTS / "driftops_bundle.zip"
    if not bundle.exists():
        return ""
    return """
      <section>
        <h2>Download</h2>
        <p><a href="driftops_bundle.zip">Download full artifact bundle (zip)</a></p>
      </section>
    """


def _checklist_section() -> str:
    items = [
        ("live_validation.json", _exists(REPORTS / "live_validation.json")),
        ("policy_gate_result.json", _exists(REPORTS / "policy_gate_result.json")),
        ("performance_metrics.json", _exists(REPORTS / "performance_metrics.json")),
        ("performance_metrics.csv", _exists(REPORTS / "performance_metrics.csv")),
        ("fairness_summary.json", _exists(REPORTS / "fairness_summary.json")),
        ("api_fairness_report.md", _exists(REPORTS / "api_fairness_report.md")),
        ("api_fairness_metrics.csv", _exists(REPORTS / "api_fairness_metrics.csv")),
        ("shap_top_features.json", _exists(REPORTS / "shap_top_features.json")),
        ("regulatory_monitor.json", _exists(REPORTS / "regulatory_monitor.json")),
        ("run_metadata.json", _exists(REPORTS / "run_metadata.json")),
        (
            "policy_registry_summary.json",
            _exists(REPORTS / "policy_registry_summary.json"),
        ),
        ("evidence_digest.json", _exists(REPORTS / "evidence_digest.json")),
        ("drift_history.json", _exists(REPORTS / "drift_history.json")),
        ("trustworthy_audit.json", _exists(REPORTS / "trustworthy_audit.json")),
        ("index.html", _exists(REPORTS / "index.html")),
        ("driftops_bundle.zip", _exists(REPORTS / "driftops_bundle.zip")),
    ]
    rows = []
    for name, present in items:
        color = "#22c55e" if present else "#ef4444"
        label = "Yes" if present else "No"
        rows.append(
            f"<tr><td>{name}</td><td><b style='color:{color}'>{label}</b></td></tr>"
        )
    return f"""
      <section>
        <h2>Evidence Checklist</h2>
        <table border="1" cellspacing="0" cellpadding="6">
          <thead><tr><th>Artifact</th><th>Present</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </section>
    """


def _integrity_section(digest: Dict[str, Any]) -> str:
    if not digest:
        return ""
    rows = []
    # Root files
    root = digest.get("root_files", {}) or {}
    for name, info in root.items():
        sha = (info or {}).get("sha256", "")
        sha_short = sha[:12] + "…" if sha else ""
        size = (info or {}).get("size_bytes", "")
        present = (info or {}).get("exists", False)
        color = "#22c55e" if present else "#ef4444"
        rows.append(
            f"<tr><td>{name}</td><td>{size}</td><td><code>{sha_short}</code></td>"
            f"<td><b style='color:{color}'>{'Yes' if present else 'No'}</b></td></tr>"
        )
    # Report files
    rep = digest.get("report_files", {}) or {}
    for name, info in rep.items():
        sha = (info or {}).get("sha256", "")
        sha_short = sha[:12] + "…" if sha else ""
        size = (info or {}).get("size_bytes", "")
        present = (info or {}).get("exists", False)
        color = "#22c55e" if present else "#ef4444"
        link = name if present else "#"
        name_html = f'<a href="{link}">{name}</a>' if present else name
        rows.append(
            f"<tr><td>{name_html}</td><td>{size}</td><td><code>{sha_short}</code></td>"
            f"<td><b style='color:{color}'>{'Yes' if present else 'No'}</b></td></tr>"
        )

    return f"""
      <section>
        <h2>Integrity Digest (SHA256)</h2>
        <table border="1" cellspacing="0" cellpadding="6">
          <thead><tr><th>File</th><th>Size (bytes)</th><th>SHA256</th><th>Present</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </section>
    """


def _drift_history_section(history) -> str:
    if not isinstance(history, list) or not history:
        return ""
    # Newest last; display most recent first
    rows = []
    for rec in reversed(history[-50:]):  # show up to 50 recent
        status = (rec.get("status") or "").upper()
        badge = _badge(status)
        ts = rec.get("ts", "")
        auroc = rec.get("auroc", "")
        ks = rec.get("ks_stat", "")
        min_auroc = rec.get("min_auroc", "")
        min_ks = rec.get("min_ks", "")
        rows.append(
            f"<tr><td>{ts}</td><td>{badge}</td><td>{auroc}</td><td>{ks}</td>"
            f"<td>{min_auroc}</td><td>{min_ks}</td></tr>"
        )
    return f"""
      <section>
        <h2>Drift History</h2>
        <table border="1" cellspacing="0" cellpadding="6">
          <thead><tr><th>Timestamp (UTC)</th><th>Status</th><th>AUROC</th><th>KS</th><th>min_auroc</th><th>min_ks</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </section>
    """


def _trustworthy_audit_section(audit: Dict[str, Any]) -> str:
    if not audit:
        return ""
    blocks: List[str] = ["<section><h2>Trustworthy Audit</h2>"]

    summary = audit.get("summary", {})
    if summary:
        blocks.append("<ul>")
        blocks.append(
            f"<li>Entries evaluated: {summary.get('entries_evaluated', 0)}</li>"
        )
        blocks.append(f"<li>Drift flags: {summary.get('drift_flags', 0)}</li>")
        blocks.append(f"<li>Max KS: {summary.get('max_ks', '—')}</li>")
        blocks.append(f"<li>Policy status: {summary.get('policy_status', '—')}</li>")
        blocks.append("</ul>")

    top_feats = audit.get("explainability", {}).get("top_features", [])
    if top_feats:
        cols = sorted({k for r in top_feats for k in r})
        thead = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
        rows = [
            "<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in cols) + "</tr>"
            for r in top_feats
        ]
        blocks.append(
            f"<table border='1' cellspacing='0' cellpadding='6'>{thead}{''.join(rows)}</table>"
        )

    blocks.append("</section>")
    return "".join(blocks)


# ------------------------------- build ---------------------------------------


def build() -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)

    live = _read_json(REPORTS / "live_validation.json")
    gate = _read_json(REPORTS / "policy_gate_result.json")
    perf = _read_json(REPORTS / "performance_metrics.json")
    shap = _read_json(REPORTS / "shap_top_features.json")
    fair = _read_json(REPORTS / "fairness_summary.json")
    regm = _read_json(REPORTS / "regulatory_monitor.json")
    rmeta = _read_json(REPORTS / "run_metadata.json")
    polsum = _read_json(REPORTS / "policy_registry_summary.json")
    digest = _read_json(REPORTS / "evidence_digest.json")
    history = _read_json(REPORTS / "drift_history.json")
    audit = _read_json(REPORTS / "trustworthy_audit.json")

    status_badge = _badge(
        (live if isinstance(live, dict) else {}).get("status", "FAIL")
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Clinical DriftOps — Reports Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 24px; }}
    h1,h2 {{ margin: 0 0 12px; }}
    h3 {{ margin: 8px 0; }}
    section {{ margin: 20px 0; }}
    .kv td {{ padding: 4px 8px; }}
    pre {{ background:#0b1221; color:#e5e7eb; padding:12px; border-radius:8px; overflow:auto; }}
    a {{ color:#2563eb; }}
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

  {_policy_table(gate if isinstance(gate, dict) else {}, perf if isinstance(perf, dict) else {})}

  <section>
    <h2>Performance</h2>
    <table class="kv" border="0" cellspacing="0" cellpadding="0">
      <tr><td>n</td><td>{(perf or {}).get('n','')}</td></tr>
      <tr><td>accuracy@0.5</td><td>{(perf or {}).get('accuracy@0.5','')}</td></tr>
      <tr><td>auroc</td><td>{(perf or {}).get('auroc','')}</td></tr>
      <tr><td>ks_stat</td><td>{(perf or {}).get('ks_stat','')}</td></tr>
    </table>
  </section>

  {_shap_section(shap if isinstance(shap, dict) else {})}
  {_fairness_section(fair if isinstance(fair, dict) else {})}
  {_regulatory_section(regm if isinstance(regm, dict) else {})}
  {_runmeta_section(rmeta if isinstance(rmeta, dict) else {})}
  {_policy_registry_section(polsum if isinstance(polsum, dict) else {})}
  {_bundle_link()}
  {_integrity_section(digest if isinstance(digest, dict) else {})}
  {_drift_history_section(history if isinstance(history, list) else [])}
  {_trustworthy_audit_section(audit if isinstance(audit, dict) else {})}
  {_checklist_section()}

</body>
</html>
"""
    target = REPORTS / "index.html"
    target.write_text(html, encoding="utf-8")
    return str(target)


if __name__ == "__main__":
    print(build())
