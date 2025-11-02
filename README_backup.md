# 🩺 Clinical DriftOps Platform • Honest AI for Healthcare
![Clinical DriftOps](https://github.com/sobcza11/clinical-driftops-platform/blob/main/assets/driftOps_pic_h.png)

<p align="center">
  <a href="https://sobcza11.github.io/clinical-driftops-platform/">
    <img src="https://img.shields.io/badge/Dashboard-Live-success?logo=github" alt="Open Dashboard">
  </a>
  <a href="https://github.com/sobcza11/clinical-driftops-platform/actions/workflows/driftops-ci.yml">
    <img src="https://github.com/sobcza11/clinical-driftops-platform/actions/workflows/driftops-ci.yml/badge.svg" alt="DriftOps CI">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License MIT">
  </a>
</p>
---

**Live Dashboard:** https://sobcza11.github.io/clinical-driftops-platform/

Governance-first MLOps framework ensuring transparent, auditable, and policy-as-code validation for clinical AI models.

> **PMI-CPMAI Aligned • MLOps • Trustworthy AI • Explainable Healthcare Models**

---

## 🌍 Overview
**Clinical DriftOps Platform** is an end-to-end **MLOps framework** for *trustworthy, explainable, and continuously validated clinical AI*.
It monitors **data drift**, **fairness**, and **model explainability**, automatically enforcing governance policies via CI/CD.
Built for clinical research using the **MIMIC-IV** dataset, aligned to **PMI-CPMAI Phases I–VI**.

---

## 🧭 CPMAI / CRISP-DM Alignment

| Phase | Description | Artifacts |
|---|---|---|
| **I. Business Understanding** | Define risk of model drift & bias in clinical ML. | `README.md`, `docs/overview.md` |
| **II. Data Understanding** | Profile MIMIC-IV labs, vitals, outcomes. | `src/data_prep.py`, EDA notebooks |
| **III. Data Preparation** | Scale & clean baseline vs. current datasets. | `data/data_prepared_*.csv`, `reports/data_prep_meta.json` |
| **IV. Modeling & Drift Detection** | PSI / KS tests + SHAP explainability. | `src/monitors/drift_detector.py`, `src/explain/shap_summary.py` |
| **V. Evaluation & Governance Gate** | Fairness, performance metrics, policy enforcement. | `src/eval/*`, `src/ops/policy_gate.py`, `reports/*` |
| **VI. Operationalization** | Live monitoring, MLflow registry, FHIR integration. | `src/ops/*`, `reports/index.html` |

---

## ⚙️ Key Features
- 🧪 **Automated Performance Audits** – AUROC, accuracy@0.5, KS via `performance_metrics.py`.
- 📈 **Data Drift Monitoring** – PSI / KS tests; thresholds in `policy.yaml`.
- 🤝 **Fairness Audits** – per-group positive-rate & parity gap (`fairness_audit.py`).
- 🩻 **Explainability via SHAP** – top features & artifact presence checks.
- 🛡️ **Policy Gate Enforcement** – single source of truth for model acceptance criteria.
- 🚀 **CI/CD** – GitHub Actions runs the pipeline and publishes a live dashboard.
- 🧾 **MLflow Logging** – metrics & artifact tracking (Phase VI expansion).

---

## 🧩 Policy Schema (`policy.yaml`)
```yaml
drift:
  psi_fail: 0.20
  ks_fail: 0.20

performance:
  min_auroc: 0.80
  # optional: min_auprc: 0.50
  # optional: max_log_loss: 0.70

fairness:
  parity_gap_fail: 0.05

explainability:
  require_shap_artifact: true
  top_features_min: 10
