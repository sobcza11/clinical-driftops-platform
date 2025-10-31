[![DriftOps CI](https://github.com/sobcza11/clinical-driftops-platform/actions/workflows/driftops-ci.yml/badge.svg)](https://github.com/sobcza11/clinical-driftops-platform/actions/workflows/driftops-ci.yml)
# 🩺 Clinical DriftOps Platform • Honest AI for Healthcare
![alt text](https://github.com/sobcza11/clinical-driftops-platform/blob/main/assets/driftOps_pic_h.png)

<p align="center">
  <a href="https://github.com/sobcza11/clinical-driftops-platform/actions/workflows/driftops-ci.yml">
    <img src="https://github.com/sobcza11/clinical-driftops-platform/actions/workflows/driftops-ci.yml/badge.svg" alt="DriftOps CI">
  </a>
  <a href="https://sobcza11.github.io/clinical-driftops-platform/">
    <img src="https://img.shields.io/badge/Open-Dashboard-success?style=for-the-badge&logo=github" alt="Click - Open the Dashboard">
  </a>
  <a href="https://sobcza11.github.io/clinical-driftops-platform/">
    <img src="https://img.shields.io/badge/Dashboard-Live-success" alt="Dashboard Live">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License MIT">
  </a>
</p>

---

**Live Dashboard:** https://sobcza11.github.io/clinical-driftops-platform/
**Status:** [![DriftOps CI](https://github.com/sobcza11/clinical-driftops-platform/actions/workflows/driftops-ci.yml/badge.svg)](https://github.com/sobcza11/clinical-driftops-platform/actions)

Governance-first MLOps framework ensuring transparent, auditable, and
policy-as-code validation for clinical AI models.

> **PMI-CPMAI Aligned • MLOps • Trustworthy AI • Explainable Healthcare Models**

---

## 🌍 Overview
**Clinical DriftOps Platform** is an end-to-end **MLOps framework** for *trustworthy, explainable, and continuously validated clinical AI*.
It monitors **data drift**, **fairness**, and **model explainability**, automatically enforcing governance policies via CI/CD.
Built for clinical research using the **MIMIC-IV** dataset, aligned to **PMI-CPMAI Phases I–VI**.

---

## 🧭 CPMAI / CRISP-DM Alignment

| Phase | Description | Artifacts |
|-------|--------------|------------|
| **I. Business Understanding** | Defined risk of model drift & bias in clinical ML. | `README.md`, `docs/overview.md` |
| **II. Data Understanding** | Profiling MIMIC-IV labs, vitals, outcomes. | `src/data_prep.py`, EDA notebooks |
| **III. Data Preparation** | Scaled & cleaned baseline vs current datasets. | `data/data_prepared_*.csv`, `reports/data_prep_meta.json` |
| **IV. Modeling & Drift Detection** | PSI / KS tests + SHAP explainability. | `src/monitors/drift_detector.py`, `src/explain/shap_summary.py` |
| **V. Evaluation & Governance Gate** | Fairness, performance metrics, policy enforcement. | `src/eval/*`, `src/ops/policy_gate.py`, `reports/*` |
| **VI. Operationalization** | *Coming next*: live monitoring, MLflow registry + FHIR integration. | (planned) |

---

## ⚙️ Key Features
- 🧪 **Automated Performance Audits** – AUROC, AUPRC, Log-loss via `performance_metrics.py`
- 📈 **Data Drift Monitoring** – PSI / KS tests; thresholds in `policy.yaml`
- 🤝 **Fairness Audits** – per-group positive-rate & parity gap (`fairness_audit.py`)
- 🩻 **Explainability via SHAP** – top features plotted + artifact presence check
- 🛡️ **Policy Gate Enforcement** – single source of truth for model acceptance criteria
- 🚀 **Continuous Integration & Deployment** – GitHub Actions runs full pipeline, publishes dashboard
- 🧾 **MLflow Logging** – automatic metrics and artifact tracking (Phase VI expansion)

---

## 🧩 Policy Schema (policy.yaml)

```yaml
drift:
  psi_fail: 0.20
  ks_fail: 0.20

performance:
  min_auroc: 0.80
  min_auprc: 0.50
  max_log_loss: 0.70

fairness:
  parity_gap_fail: 0.05

explainability:
  require_shap_artifact: true
  top_features_min: 10


## 🚀 Phase VI — Operationalization (Coming Soon)

> Building toward live, explainable, continuously governed clinical AI

| Capability | Description | Target Artifact |
|-------------|--------------|----------------|
| **Real-Time MLflow Registry** | Transition from file-based tracking to MLflow’s tracking server for experiment lineage & model versioning. | `mlruns/registry.json` |
| **Continuous Validation API** | REST endpoint for on-demand drift/fairness scoring across live inference data. | `src/api/validation_server.py` |
| **FHIR Integration** | Stream ICU vitals + labs into MIMIC-compatible pipelines for real-time monitoring. | `src/connectors/fhir_stream.py` |
| **Alerting & Human-in-Loop Review** | Slack/Teams notifications when policy gate fails, with clinician acknowledgment workflow. | `src/ops/alerting.py` |
| **Governance Dashboard v2** | Aggregate multi-model results into a single transparent clinical governance panel. | `reports/dashboard_v2.html` |

🧭 *Phase VI aligns to PMI-CPMAI’s Operationalization & Monitoring phases — turning governance into live assurance.*
