# CIF Credit Intelligence

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?logo=open-source-initiative&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6F00?logo=python&logoColor=white)

**Credit decision support platform for the CIF ecosystem.** Risk scoring, thin-file management, and comprehensive methodological audit.

> Plateforme d'aide à la décision de crédit pour les SFD de la CIF. Scoring de risque, gestion des Thin-File, audit méthodologique complet.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [API Endpoint](#api-endpoint)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Technical Stack](#technical-stack)
- [CIF Compliance](#cif-compliance)
- [Validation](#validation)
- [Project Status](#project-status)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

| | |
|---|---|
| **Ecosystem** | CIF / DigiCoop-WA+ |
| **Language** | Python 3.11 |
| **License** | Apache 2.0 |
| **Status** | Production |
| **API** | https://cif-credit-intelligence.onrender.com/docs |

A credit scoring platform engineered as a financial intelligence infrastructure, following professional consulting firm standards: **reproducible, tested, instrumented, deployable**.

The prototype's ROC-AUC of **0.94** (`cif_credit_official:1.0.1`) is an experimental result on synthetic data—not a CIF benchmark. This repository contains the engineering-grade version, ready to ingest real CIF data through the V1.1 audit protocol.

---

## Features

- **Scoring Models**
  - CIF Pilot (Synthetic): 25 features, experimental validation
  - Lending Club Champion: 18 features, public-data benchmark

- **Infrastructure**
  - Stateless FastAPI serving with model registry and promotion/rollback
  - Versioned PostgreSQL 16 schema with Alembic migrations
  - Dagster orchestration for reproducible pipelines

- **Quality Assurance**
  - 143 unit + integration tests (all in CI)
  - 87% code coverage
  - Anti-leakage detection (nominal + structural)
  - Temporal split enforcement

- **Monitoring & Observability**
  - Drift detection (Evidently + PSI)
  - Prometheus metrics + Grafana dashboards
  - Audit trail (`jsonb`, PostgreSQL)

- **Deployment Ready**
  - Render (current: single instance)
  - Kubernetes manifests (K3s + Oracle Cloud)
  - Infrastructure-as-code (Terraform)

---

## API Endpoint

**Production:** https://cif-credit-intelligence.onrender.com/docs

Interactive Swagger UI, directly testable in-browser.

| Model | Endpoint | Features | Data Source |
|---|---|---|---|
| **CIF Pilot** | `/v1/predict` | 25 | Synthetic |
| **Lending Club** | `/v1/lending-club/score` | 18 | Public real data |

---

## Installation

### Requirements

```bash
- Python 3.11
- pip / poetry
- PostgreSQL 16 (optional; SQLite for dev)
```

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Sow221/cif-credit-intelligencedescription.git
cd cif-credit-intelligencedescription

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate           # Unix/macOS
.venv\Scripts\activate              # Windows

# 3. Install dependencies
pip install -e ".[dev]"
```

---

## Quick Start

### Synthetic Data Pipeline

```bash
# Generate synthetic data
cif-generate

# Train model (MLflow tracking)
cif-train

# Evaluate
cif-evaluate
```

### Full Stack (Docker)

```bash
docker compose -f infra/docker-compose.yml up -d
```

| Service | URL |
|---|---|
| **MLflow** | http://localhost:5000 |
| **Dagster** | http://localhost:3000 |
| **Grafana** | http://localhost:3001 |
| **API** | http://localhost:8000/docs |

### Public Data Validation (Lending Club)

```bash
pip install -e ".[dev,data]"
make data-download      # Kaggle CLI
make pipeline           # Ingest + benchmark
cif-replay-monitoring   # Monitor over 24-month history
```

---

## Architecture

### Repository Structure

```text
src/
├── api/                    FastAPI application
├── config/                 Pydantic + Hydra settings
├── data/                   Synthetic generator
├── evaluation/             Metrics, calibration, fairness
├── features/
│   ├── definitions/        One file per feature (25 total)
│   └── builder.py          Centralized computation
├── models/                 XGBoost training + model_card
├── monitoring/             Drift (Evidently), Prometheus
├── services/               decision_engine, audit_service
├── cli/                    CLI commands (cif-*)
└── utils/                  Logging, reproducibility

migrations/                 Alembic (PostgreSQL schema)
pipelines/                  Dagster definitions
conf/                       Hydra YAML config
data/                       Datasets, model cards
deploy/                     Deployment artifacts
docker/                     Dockerfiles
infra/                      Local stack (Docker Compose)
k8s/ terraform/             K3s + IaC (ready, inactive)
tests/                      Unit + integration tests
docs/                       ADRs, runbooks, validation
```

---

## Technical Stack

| Layer | Tool | Purpose |
|---|---|---|
| **Language** | Python 3.11 (mypy, ruff) | Type safety, code quality |
| **Config** | Pydantic v2 + Hydra | Settings & schema management |
| **Orchestration** | Dagster | Asset versioning, lineage |
| **ML Registry** | MLflow (Postgres + MinIO) | Model tracking, promotion |
| **Data Quality** | Pandera + Pydantic v2 | Blocking contracts |
| **Data Versioning** | DVC | Reproducible pipelines |
| **Monitoring** | Evidently + Prometheus + Grafana | Drift, observability |
| **Serving** | FastAPI | HTTP API, inference |
| **Database** | PostgreSQL 16 + Alembic | Audit trail, versioning |
| **CI/CD** | GitHub Actions | Lint → test → build → promote |
| **Deployment** | Render (current) / K8s | Container orchestration |

---

## CIF Compliance

Implements **§M07 / §64 / §65 / §93** of the CIF Digital Platform specification:

- ✅ Model / Policy / Workflow separation
- ✅ Human-in-the-loop decision making
- ✅ Model governance
- ✅ Monitoring & rollback

### Methodological Validation

All CIF protocol guarantees are **implemented in code**, not just documented.

#### Temporal Split Enforcement

**File:** `src/models/train.py` → `temporal_split`

Training never uses random splits; cutoff follows temporal order.

**Locked by:** `tests/unit/test_temporal_split.py`

#### Anti-Leakage Guards (2 Levels)

**Files:** `src/features/validate.py` + `src/features/builder.py`

1. **Nominal blocklist:** Any leak-named feature (e.g., `p_default_true`) triggers a blocking error and a 422 API response.
2. **Structural guard:** Features correlating > 0.75 with the target are blocked. Detects semantic leakage (e.g., `historical_default_rate` ≈ 0.94 correlation—never caught by name alone).

**Locked by:** `tests/unit/test_leakage.py` + `tests/integration/test_api.py`

#### Feature-Set = Official Model

**File:** `src/features/builder.py`

The 25 `cifci` features (profile, income, loan aggregations, derived ratios) are reconstructed as a single source of truth with contracts.

#### Model Hyperparameters

```text
max_depth=4
learning_rate=0.03
n_estimators=300
```

---

## Validation

### Public Real Data (Lending Club)

**Out-of-time protocol** (see ADRs for full details):

- Split by origination date
- Logistic baseline + XGBoost (monotonicity constraints)
- Isotonic calibration on validation
- Champion retained only if AUC gap is **statistically significant**

**Result:** Validates **methodology**, not CIF portfolio performance.

### Monitoring Replay

Simulates 24-month production surveillance over real history:

- PSI drift: available immediately
- Real performance: 12-month holdout only (never touched in training)

**Finding:** 4 months trigger a PSI alert on one variable; performance remains stable across the period.

---

## Project Status

| Phase | Objective | Status | Reference |
|---|---|---|---|
| **1** | Infrastructure | ✅ Complete, Production | This repo |
| **2** | Public data validation | ✅ Complete | `docs/validation/` |
| **3** | CIF real data (shadow mode) | 🔄 Planned | – |

### Code Quality

- **143 tests** (unit + integration, all in CI)
- **Coverage:** 87%
- **CLI:** all 11 commands tested via `click.testing.CliRunner`

### Delivery Timeline

| Week | Deliverables | Status |
|---|---|---|
| 1 | Foundation (25 contracts, Pydantic settings) | ✓ `971deeb` |
| 2 | API (`/v1/predict`, JWT, rate limiting) | ✓ `030ece9` |
| 3 | Database (Alembic, PostgreSQL 16 schema) | ✓ Complete |
| 4 | MLflow + Monitoring (train, drift) | ✓ `117e8a6` |
| 5 | Kubernetes + Terraform (manifests, IaC) | ✓ `637e402` |
| 6 | CI/CD Canary (GitHub Actions) | ✓ `61c34e7` |

---

## Documentation

| Resource | Path | Purpose |
|---|---|---|
| **Data Governance** | `data/README.md` | Dataset management, Kaggle setup |
| **Deployment** | `deploy/huggingface/README.md` | Containerized API, security |
| **Architecture** | `docs/adr/` | ADRs, design rationale |
| **Operations** | `docs/runbooks/` | Troubleshooting, rollback |
| **Validation** | `docs/validation/` | Benchmark results, monitoring |

---

## License

**Apache License 2.0**

Developed for the CIF / DigiCoop-WA+ ecosystem.

---

**Questions?** See the [full documentation](https://github.com/Sow221/cif-credit-intelligencedescription/tree/main/docs) or open an [issue](https://github.com/Sow221/cif-credit-intelligencedescription/issues).
