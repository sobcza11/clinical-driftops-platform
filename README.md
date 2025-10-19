<h1 align="center">🏥 Clinical DriftOps Platform</h1>
<h3 align="center"><i>PMI-CPMAI™-Aligned MLOps for Safe, Explainable Clinical AI</i></h3>

<p align="center">
  <img src="assets/driftOps_pic_h.png" alt="Clinical DriftOps Overview" width="600"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PMI--CPMAI™-blueviolet" />
  <img src="https://img.shields.io/badge/Governance-HIPAA%20%7C%20FDA%20GMLP%20%7C%20EU%20AI%20Act-success" />
  <img src="https://img.shields.io/badge/Monitoring-Evidently%20PSI%20%7C%20KS-informational" />
  <img src="https://img.shields.io/badge/Explainability-SHAP%20%7C%20Permutation%20Importance-lightgrey" />
  <img src="https://img.shields.io/badge/Tracking-MLflow%20Lineage%20%7C%20Artifacts-orange" />
  <img src="https://img.shields.io/badge/Language-Python%203.10+-blue" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

---

## 🧠 Objective
Establish a **repeatable, auditable, and ethical MLOps framework** to detect and mitigate **data drift, model drift, & bias** across clinical AI use cases (e.g., sepsis risk, readmission, medication adherence) while preserving **trust, transparency, & regulatory compliance**.

---

## 📌 Highlights
- **PMI-CPMAI™ aligned** lifecycle with explicit deliverables per phase (I → VI)
- **Real-time monitoring:** PSI, KS, latency, and prediction errors
- **Explainability:** SHAP + Permutation Importance, clinician-readable summaries
- **Governance:** MLflow lineage, model cards, Policy-as-Code monitors
- **Compliance:** HIPAA, FDA GMLP, EU AI Act, human-in-loop safeguards

---

## 🧩 CPMAI Phase Map (I → VI)

| Phase | Objective | Key Outputs |
| :--- | :--- | :--- |
| **I. Business Understanding** | Define clinical problem, KPIs, stakeholders, ethics. | Charter · KPI Matrix · Risk Register |
| **II. Data Understanding** | Profile MIMIC-IV / FHIR sources; identify drift & bias. | Evidently Baseline Report · Data Dictionary |
| **III. Data Preparation** | PII-safe transforms, feature store, versioning. | `data_prep.py` · Metadata JSON |
| **IV. Model Development** | Drift/bias detection + explainability; containerization. | `driftops_service.py` · Dockerfile |
| **V. Model Evaluation** | Validate reliability & trustworthiness. | “Trustworthy AI Audit v1.0” PDF/MD |
| **VI. Operationalization** | CI/CD · MLflow · live monitors · HITL retraining. | MLflow Registry · Policy Sentinel · Dashboards |

---

## 📊 Key Performance Indicators

| KPI | Target | Measurement | Owner |
| :--- | ---: | :--- | :--- |
| False-alert reduction | ≥ 20 % vs baseline | Alert log analysis | Clinical Lead |
| Drift detection latency | ≤ 3 days | PSI/KS pipeline | Data Science Lead |
| Clinician trust score | ≥ 8 / 10 | Surveys · Focus Groups | Compliance Officer |
| Model AUC stability | ≤ 5 % QoQ decay | Validation dashboard | Data Ops Engineer |

---

## ⚖️ Compliance & Ethics
- **HIPAA:** de-identification; no PHI in repo  
- **FDA GMLP:** lineage · reproducibility · audit controls  
- **EU AI Act:** documentation · risk classification · monitoring  
- **NIST SP 800-53:** security baseline for pipelines  
- **Human-in-Loop (HITL):** approval at critical decision nodes  
- **Audit Trails:** MLflow artifacts + signed reports  

---

## 📂 Repository Structure
clinical-driftops-platform/
├─ assets/
│ └─ driftOps_pic_h.png # banner image
├─ data/ # sample or synthetic schemas
├─ notebooks/ # EDA · drift baseline · bias analysis
├─ src/
│ ├─ data_prep.py # PII-safe transforms + lineage
│ ├─ driftops_service.py # APIs for drift / bias / explain
│ ├─ monitors/ # PSI · KS · latency · error monitors
│ └─ explain/ # SHAP + Permutation Importance
├─ ops/
│ ├─ ci/ # GitHub Actions / CI workflows
│ ├─ mlflow/ # experiment registry config
│ └─ policy/ # Policy-as-Code sentinels
├─ reports/
│ ├─ trustworthy_ai_audit_v1.md # Phase V artifact
│ └─ kpi_dashboard_links.md
├─ LICENSE
└─ README.md

---

## 🚀 Quickstart

```bash
# Clone
git clone https://github.com/sobcza11/clinical-driftops-platform.git
cd clinical-driftops-platform

# (Optional) create venv
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux

# Install
pip install -r requirements.txt

# Run baseline drift check
python -m src.monitors.run_baseline --config configs/baseline.yaml

# Launch MLflow UI
mlflow ui


🧮 Explainability & Fairness

SHAP (tree/kernel) for local + global attribution

Permutation Importance as model-agnostic backup

Fairness Slices across age, gender, race + bias remediation playbook

🧭 Roadmap

Current: Phase II – Data Understanding (phase-ii-data-understanding)

Next: Streamlit / Grafana dashboards for live drift monitoring

Planned: GenAI Compliance Sentinel for RAG-based FDA/HIPAA/EU AI Act diff alerts

👤 Credits

Rand Sobczak Jr., PMI-CPMAI™
Project Lead · Clinical MLOps Architect