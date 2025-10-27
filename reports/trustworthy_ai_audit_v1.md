# Trustworthy AI Audit v1.0 (Phase V)

_Generated: 2025-10-24 21:30 UTC_

## Overview
- **Project:** Clinical DriftOps Platform
- **Scope:** Phase V – Model Evaluation (drift, explainability, fairness)

## 1) Data Drift Summary
- Source: `reports/drift_metrics.csv`
- Features flagged for drift (heuristic): **0** (PSI ≥ 0.2)

| feature    |       psi |   ks_stat |   ks_pvalue | drift_flag   |
|:-----------|----------:|----------:|------------:|:-------------|
| anchor_age | 0.0809946 | 0.0412089 |    0.869805 | False        |
| lactate    | 0.0488398 | 0.035913  |    0.948722 | False        |
| creatinine | 0.0311234 | 0.0426602 |    0.842153 | False        |
| wbc        | 0.0276683 | 0.038771  |    0.910816 | False        |

- Full Evidently report: `reports/data_drift_small_demo.html`

## 2) Explainability (SHAP)
![SHAP Top Features](reports/shap_top_features.png)

## 3) Fairness Audit
- Max absolute disparity (heuristic): **0.020** (**threshold 0.05**)

| group   |   n |   positive_rate |   disparity |
|:--------|----:|----------------:|------------:|
| F       | 161 |       0.0186335 |  -0.0202266 |
| M       | 225 |       0.0533333 |   0.0144732 |

### Fairness Report (Markdown)

# Fairness Audit

**Group column:** `gender`

**Overall positive rate:** 0.0389

| group   |   n |   positive_rate |   disparity |
|:--------|----:|----------------:|------------:|
| F       | 161 |       0.0186335 |  -0.0202266 |
| M       | 225 |       0.0533333 |   0.0144732 |


## 4) Governance Notes
- HIPAA-aligned (de-identified data), FDA GMLP lineage & reproducibility, EU AI Act monitoring.
- Human-in-the-loop review recommended for flagged drift or fairness gaps.
