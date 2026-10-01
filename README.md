<p align="center">
  <strong>CIF Credit Intelligence</strong><br/>
  <em>Credit decision support platform for the CIF ecosystem</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?logo=open-source-initiative&logoColor=white" alt="License">
  <img src="https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/XGBoost-ML-FF6F00?logo=python&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Status-Production-success" alt="Status">
</p>

<p align="center">
  <strong>Risk scoring, thin-file management, comprehensive methodological audit</strong><br/>
  <em>Plateforme d'aide à la décision de crédit pour les SFD de la CIF</em>
</p>

---

## Table of Contents

| Section | Link |
|:-------:|:---|
| Overview | [See below](#overview) |
| Features | [See below](#features) |
| API | [See below](#api-endpoint) |
| Installation | [See below](#installation) |
| Quick Start | [See below](#quick-start) |
| Architecture | [See below](#architecture) |
| Technical Stack | [See below](#technical-stack) |
| Compliance | [See below](#cif-compliance) |
| Validation | [See below](#validation) |
| Status | [See below](#project-status) |
| Documentation | [See below](#documentation) |

---

## Overview

A credit scoring platform engineered as a financial intelligence infrastructure following professional consulting firm standards: **reproducible, tested, instrumented, deployable**.

<br/>

| Key Attribute | Value |
|:---|:---|
| **Ecosystem** | CIF / DigiCoop-WA+ |
| **Language** | Python 3.11 |
| **License** | Apache 2.0 |
| **Status** | Production |
| **API** | https://cif-credit-intelligence.onrender.com/docs |

<br/>

> **Note:** The prototype's ROC-AUC of **0.94** is an **experimental result on synthetic data**, not a CIF benchmark. This repository contains the **engineering-grade version**, ready to ingest real CIF data.

---

## Features

<details>
  <summary><strong>Scoring Models</strong></summary>

  * **CIF Pilot** (Synthetic)
    * 25 features
    * Experimental validation
    * Ready for real CIF data

  * **Lending Club Champion** (Public Real Data)
    * 18 features
    * Validated on Kaggle dataset
    * Methodology proof-of-concept

</details>

<details>
  <summary><strong>Infrastructure</strong></summary>

  * **API** - Stateless FastAPI with model registry and promotion/rollback
  * **Storage** - Versioned PostgreSQL 16 schema (Alembic migrations)
  * **Orchestration** - Dagster for reproducible pipelines
  * **Tracking** - MLflow + MinIO for model versioning

</details>

<details>
  <summary><strong>Quality Assurance</strong></summary>

  * 143 unit + integration tests (all in CI)
  * 87% code coverage (mypy + ruff strict)
  * Anti-leakage detection (nominal + structural)
  * Temporal split enforcement (no random splits)

</details>

<details>
  <summary><strong>Monitoring & Observability</strong></summary>

  * **Drift Detection** - Evidently + PSI metrics
  * **Metrics** - Prometheus + Grafana dashboards
  * **Audit** - Complete PostgreSQL jsonb trail
  * **Validation** - 24-month monitoring replay

</details>

<details>
  <summary><strong>Deployment Ready</strong></summary>

  * **Current** - Render (single instance, zero-cost)
  * **Next** - Kubernetes (K3s + Oracle Cloud)
  * **Infrastructure** - Terraform IaC (ready)
  * **Design** - Stateless API for horizontal scaling

</details>

---

## API Endpoint

<p align="center">
  <strong>Live: https://cif-credit-intelligence.onrender.com/docs</strong><br/>
  <em>Interactive Swagger UI, directly testable</em>
</p>

<br/>

| Model | Endpoint | Features | Data |
|:---|:---|:---:|:---|
| **CIF Pilot** | `/v1/predict` | 25 | Synthetic |
| **Lending Club** | `/v1/lending-club/score` | 18 | Public Real |

---

## Installation

### Requirements

```bash
Python 3.11+
pip or poetry
PostgreSQL 16 (optional: SQLite for dev)
```

### Setup

**Step 1: Clone repository**

```bash
git clone https://github.com/Sow221/cif-credit-intelligencedescription.git
cd cif-credit-intelligencedescription
```

**Step 2: Create virtual environment**

```bash
python -m venv .venv

# On Unix/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

**Step 3: Install dependencies**

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### Option 1: Synthetic Data Pipeline

```bash
cif-generate      # Generate synthetic data
cif-train         # Train model (MLflow tracking)
cif-evaluate      # Evaluate model
```

### Option 2: Full Stack (Docker)

```bash
docker compose -f infra/docker-compose.yml up -d
```

Services will be available at:

| Service | URL |
|:---|:---|
| **MLflow** | http://localhost:5000 |
| **Dagster** | http://localhost:3000 |
| **Grafana** | http://localhost:3001 |
| **API** | http://localhost:8000/docs |

### Option 3: Public Data Validation

```bash
pip install -e ".[dev,data]"
make data-download      # Download from Kaggle
make pipeline           # Ingest + benchmark
cif-replay-monitoring   # Monitoring replay (24 months)
```

---

## Architecture

### Repository Structure

```text
src/
├── api/                    FastAPI application
├── config/                 Pydantic + Hydra configuration
├── data/                   Synthetic data generator
├── evaluation/             Metrics, calibration, fairness
├── features/
│   ├── definitions/        25 feature contracts
│   └── builder.py          Centralized computation
├── models/                 XGBoost + model_card
├── monitoring/             Drift (Evidently), Prometheus
├── services/               decision_engine, audit_service
├── cli/                    CLI commands (cif-*)
└── utils/                  Logging, reproducibility

migrations/                 Alembic (PostgreSQL schema versioning)
pipelines/                  Dagster assets and jobs
conf/                       Hydra YAML configuration
data/                       Datasets, model cards
deploy/                     Deployment artifacts
docker/                     Dockerfiles
infra/                      Docker Compose stack
k8s/ terraform/             Kubernetes + IaC (ready)
tests/                      Unit and integration tests
docs/                       ADRs, runbooks, validation reports
```

---

## Technical Stack

| Layer | Tool | Purpose |
|:---|:---|:---|
| **Language** | Python 3.11 (mypy strict, ruff) | Type safety, code quality |
| **Configuration** | Pydantic v2 + Hydra | Settings management |
| **Orchestration** | Dagster | Versioned assets, lineage |
| **ML Registry** | MLflow (Postgres + MinIO) | Model tracking, promotion |
| **Data Quality** | Pandera + Pydantic v2 | Contracts, validation |
| **Data Versioning** | DVC | Reproducible pipelines |
| **Monitoring** | Evidently + Prometheus + Grafana | Drift, observability |
| **Serving** | FastAPI | HTTP API, inference |
| **Database** | PostgreSQL 16 + Alembic | Audit trail, versioning |
| **CI/CD** | GitHub Actions | Lint → test → build → promote |
| **Deployment** | Render / Kubernetes | Container orchestration |

---

## CIF Compliance

Implements **M07 / 64 / 65 / 93** of the CIF Digital Platform specification:

* Model / Policy / Workflow separation
* Human-in-the-loop decision making
* Model governance and versioning
* Monitoring and rollback

### Methodological Validation

All CIF protocol guarantees are **implemented in code**, not just documented.

<details>
  <summary><strong>Temporal Split Enforcement</strong></summary>

  **File:** `src/models/train.py` → `temporal_split`

  Training never uses random splits; cutoff follows temporal order.

  **Locked by:** `tests/unit/test_temporal_split.py`

</details>

<details>
  <summary><strong>Anti-Leakage Guards (2 Levels)</strong></summary>

  **Files:** `src/features/validate.py` + `src/features/builder.py`

  1. **Nominal blocklist** - Any leak-named feature (e.g., `p_default_true`) triggers blocking error + 422 response
  2. **Structural guard** - Features correlating > 0.75 with target are blocked

  Example: `historical_default_rate` (correlation ≈ 0.94) never caught by name alone.

  **Locked by:** `tests/unit/test_leakage.py` + `tests/integration/test_api.py`

</details>

<details>
  <summary><strong>Feature-Set = Official Model</strong></summary>

  **File:** `src/features/builder.py`

  The 25 `cifci` features are reconstructed as a single source of truth with contracts.

</details>

<details>
  <summary><strong>Model Hyperparameters</strong></summary>

  ```text
  max_depth=4
  learning_rate=0.03
  n_estimators=300
  ```

</details>

---

## Validation

<details>
  <summary><strong>Public Real Data (Lending Club)</strong></summary>

  **Out-of-time protocol:**

  * Split by origination date
  * Logistic baseline + XGBoost (monotonicity constraints)
  * Isotonic calibration on validation
  * Champion retained only if AUC gap is statistically significant

  **Result:** Validates methodology, not CIF portfolio performance.

</details>

<details>
  <summary><strong>Monitoring Replay (24 Months)</strong></summary>

  Simulates production surveillance:

  * **PSI drift** - Available immediately
  * **Real performance** - 12-month holdout (never touched in training)

  **Finding:** 4 months trigger PSI alert; performance stable across period.

</details>

---

## Project Status

| Phase | Objective | Status | Reference |
|:---:|:---|:---:|:---|
| **1** | Infrastructure | Complete | This repository |
| **2** | Public data validation | Complete | `docs/validation/` |
| **3** | CIF real data (shadow mode) | Planned | – |

### Code Quality Metrics

| Metric | Value |
|:---|:---:|
| **Tests** | 143 (unit + integration) |
| **Coverage** | 87% |
| **CI Status** | All passing |

### Delivery Timeline

| Week | Deliverables | Status |
|:---:|:---|:---:|
| 1 | Foundation (25 contracts, Pydantic) | Complete (`971deeb`) |
| 2 | API (`/v1/predict`, JWT, rate limiting) | Complete (`030ece9`) |
| 3 | Database (Alembic, PostgreSQL 16) | Complete |
| 4 | MLflow + Monitoring (train, drift) | Complete (`117e8a6`) |
| 5 | Kubernetes + Terraform (manifests, IaC) | Complete (`637e402`) |
| 6 | CI/CD Canary (GitHub Actions) | Complete (`61c34e7`) |

---

## Documentation

| Resource | Path | Purpose |
|:---|:---|:---|
| **Data Governance** | `data/README.md` | Dataset management, Kaggle setup |
| **Deployment** | `deploy/huggingface/README.md` | Containerized API, security |
| **Architecture Decisions** | `docs/adr/` | ADRs and design rationale |
| **Operations** | `docs/runbooks/` | Troubleshooting and rollback |
| **Validation Reports** | `docs/validation/` | Benchmarks and monitoring |

---

## License

<p align="center">
  <strong>Apache License 2.0</strong><br/>
  Developed for the <strong>CIF / DigiCoop-WA+</strong> ecosystem
</p>

<br/>

<p align="center">
  <a href="https://github.com/Sow221/cif-credit-intelligencedescription/tree/main/docs"><strong>Full Documentation</strong></a> · 
  <a href="https://github.com/Sow221/cif-credit-intelligencedescription/issues"><strong>Issues</strong></a> · 
  <a href="https://github.com/Sow221/cif-credit-intelligencedescription/discussions"><strong>Discussions</strong></a>
</p>

---

<p align="center">
  <em>Made with care for credit risk excellence</em>
</p>
