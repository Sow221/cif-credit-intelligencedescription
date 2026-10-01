# CIF Credit Intelligence

**Credit decision support platform for the CIF ecosystem.** Risk scoring, thin-file management, and comprehensive methodological audit.

> Plateforme d'aide à la décision de crédit pour les SFD de la CIF. Scoring de risque, gestion des Thin-File, audit méthodologique complet.

---

## Overview

| | |
|---|---|
| **Ecosystem** | CIF / DigiCoop-WA+ |
| **Language** | Python 3.11 |
| **License** | Apache 2.0 |
| **API Status** | Production |
| **Endpoint** | https://cif-credit-intelligence.onrender.com/docs |

A credit scoring platform engineered as a financial intelligence infrastructure, following the standards of a professional consulting firm: reproducible, tested, instrumented, deployable.

The prototype’s ROC-AUC of 0.94 (`cif_credit_official:1.0.1`) is an experimental result on synthetic data, not a CIF benchmark. This repository contains the engineering-grade version, ready to ingest real CIF data through the V1.1 audit protocol.

---

## API — Production

**https://cif-credit-intelligence.onrender.com/docs**

Interactive Swagger documentation, directly testable in the browser. Two models are served:

| Model | Endpoint | Features | Validation |
|---|---|---|---|
| CIF Pilot (Synthetic) | `/v1/predict` | 25 | Experimental |
| Lending Club Champion | `/v1/lending-club/score` | 18 | Public data benchmark |

See `docs/validation/lending-club-benchmark.md` for validation details.

### Deployment Architecture

The current deployment (Render, single free instance) is intentionally lightweight—not a design limit.

The architecture is built for scale from day one:

- Stateless API; model loaded once, no session state
- Versioned model registry with promotion and rollback (`src/models/promotion.py`)
- Kubernetes manifests with replicas (`k8s/deployment.yaml`)
- Infrastructure-as-code ready (`terraform/`) for dedicated cluster

Scaling is a deployment target change, not a rewrite.

---

## Repository Structure

```text
src/
├── api/                    FastAPI application (routes, schemas, middleware)
├── config/                 Pydantic settings + Hydra schema
├── data/                   Synthetic data generator
├── evaluation/             Metrics, calibration, bootstrap, robustness, fairness
├── features/
│   ├── definitions/        One file per feature + contract (name, version, bounds, owner, SLA)
│   └── builder.py          Centralized computation (no duplicated formulas)
├── models/                 XGBoost training + model_card.py
├── monitoring/             Drift detection (Evidently), Prometheus metrics
├── services/               decision_engine, confidence, predictor, audit_service
├── cli/                    CLI commands (cif-*)
└── utils/                  Structured logging, reproducibility

migrations/                 Alembic (PostgreSQL schema, versioned)
pipelines/                  Dagster definitions (assets, jobs, schedules)
conf/                       Hydra configuration (YAML)
data/                       Datasets (unversioned), model cards, baseline drift
deploy/                     Deployment artifacts (models baked into image)
docker/                     Dockerfiles (api, dagster, mlflow)
infra/                      Local stack (Docker Compose, Prometheus, Grafana, Alertmanager)
k8s/ terraform/             Scale target (K3s + Oracle Cloud Always Free), ready but inactive
scripts/                    Operations scripts (deploy, data download)
tests/                      Unit and integration tests
docs/                       Charter, ADRs, runbooks, validation reports
reports/                    Generated publishable reports (metrics, backtests)
```

---

## Technical Stack

| Layer | Tool | Purpose |
|---|---|---|
| Language | Python 3.11 + strict typing (mypy, ruff) | Code quality, maintainability |
| Configuration | Pydantic v2 + Hydra | Settings management |
| Orchestration | Dagster | Versioned assets, lineage tracking |
| Model Tracking | MLflow (Postgres + MinIO) | Registry, experiment tracking |
| Data Quality | Pandera + Pydantic v2 | Blocking contracts on data shape |
| Data Versioning | DVC | Reproducible pipelines |
| Monitoring | Evidently + Prometheus + Grafana | Drift detection, observability |
| Serving | FastAPI | HTTP API, model inference |
| Database | PostgreSQL 16 + Alembic | Audit trail, schema versioning |
| CI/CD | GitHub Actions | Lint → mypy → test → build → promote |
| Deployment | Render (current) / Kubernetes | Container orchestration |

---

## Quick Start

### Local Installation

```bash
python -m venv .venv
source .venv/bin/activate         # Unix/macOS
.venv\Scripts\activate            # Windows

pip install -e ".[dev]"

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
| MLflow | http://localhost:5000 |
| Dagster | http://localhost:3000 |
| Grafana | http://localhost:3001 |
| API Scoring | http://localhost:8000/docs |

---

## CIF Compliance

This repository implements requirements §M07 / §64 / §65 / §93 of the CIF Digital Platform specification:

- Model / Policy / Workflow separation
- Human-in-the-loop decision making
- Model governance
- Monitoring and rollback

See `docs/` and `docs/traçabilité.md` for complete details.

### Methodological Validation

CIF protocol validity guarantees are implemented in code, not just documented.

#### Temporal Split

**File:** `src/models/train.py` → `temporal_split`

Training never uses random splits. Cutoff follows temporal order (proxy: `customer_id` on synthetic data; real data: `application_date`).

**Locked by:** `tests/unit/test_temporal_split.py`

#### Anti-Leakage Guards

**Files:** `src/features/validate.py` + `src/features/builder.py`

Two levels of protection:

1. **Nominal blocklist.** Any feature named as a leak (e.g., `p_default_true`) triggers a blocking error at feature engineering and a 422 API response. The generator (`src/data/synthetic.py`) no longer diffuses such variables into the client table.

2. **Structural guard.** `assert_no_correlation_leakage`: any feature correlating > 0.75 with the target is blocking. Detects semantic leakage a column name alone would miss. Historical example: `historical_default_rate` derived from `loan_status` calculated from the target (correlation ≈ 0.94), never caught by nominal blocklist alone.

**Locked by:** `tests/unit/test_leakage.py` and `tests/integration/test_api.py`

#### Feature Set Equals Official Calibrated Model

**File:** `src/features/builder.py`

The 25 features from the `cifci` reference (profile/income, loan aggregations, derived ratios) are reconstructed as a single source of truth, with contracts (`src/features/definitions/`) and tests.

#### Model Hyperparameters

Aligned with the officially calibrated version:

```text
max_depth=4
learning_rate=0.03
n_estimators=300
```

---

## Validation

### Public Real Data (Lending Club)

```bash
pip install -e ".[dev,data]"
make data-download      # Kaggle CLI; see data/README.md
make pipeline           # ingest (Pandera validation) + benchmark (baseline vs XGBoost)
cif-replay-monitoring   # monitoring replay over 24-month history (drift + delayed performance)
```

**Out-of-time protocol** (documented in ADRs):

- Split by origination date
- Logistic baseline
- XGBoost with monotonicity constraints (Optuna, temporal CV)
- Isotonic calibration on validation fold
- Test evaluated once; champion retained only if AUC gap is statistically significant

These results validate the methodology, not performance on the CIF portfolio.

**Monitoring replay** (`docs/validation/monitoring-replay.md`): simulates production surveillance over 24 months of real history. PSI drift is available immediately; real performance only on the 12-month holdout test (never touched in training/calibration, with the 36-month lag a real deployment would have seen).

Non-engineered result: 4 months trigger a PSI alert on one variable; performance remains stable across the period.

---

## Project Status

| Phase | Objective | Status | Reference |
|---|---|---|---|
| 1 | Infrastructure | Complete, production | This repository |
| 2 | Validation on public real data | Complete | API documentation + `docs/validation/` |
| 3 | CIF real data protocol (shadow mode) | Planned | – |

### Code Quality

- **143 tests** passing (unit + integration, all run in CI)
- **Coverage:** 87%
- **CLI coverage:** all 11 commands tested via `click.testing.CliRunner`

### Delivery Timeline

| Week | Deliverables | Status |
|---|---|---|
| 1 — Foundation | Repository structure, 25 feature contracts, Pydantic settings | ✓ Complete (`971deeb`) |
| 2 — API | `/v1/predict`, `/v1/auth/token`, JWT HS256, rate limiting, `X-Request-ID` | ✓ Complete (`030ece9`) |
| 3 — Database | Alembic migrations, audit service, PostgreSQL 16 schema (customers, predictions, audit_log, model_versions) | ✓ Complete |
| 4 — MLflow & Monitoring | Model training, model card, drift reporting | ✓ Complete (`117e8a6`) |
| 5 — Kubernetes & Terraform | Manifests (deployment, service, ingress), IaC ready | ✓ Complete (`637e402`) |
| 6 — CI/CD & Canary | GitHub Actions workflow, deployment script | ✓ Complete (`61c34e7`) |

### Database

Schema (customers, predictions, audit_log, model_versions) is versioned via `migrations/` (Alembic, target PostgreSQL 16).

- **Local/Test:** SQLite (`CIF_DATABASE__URL=sqlite+pysqlite:///audit.db`)
- **Production:** Native PostgreSQL `jsonb`

Audit is enabled automatically when `CIF_DATABASE__URL` is set.

---

## Documentation

| Resource | Path | Purpose |
|---|---|---|
| Data governance | `data/README.md` | Dataset management, Kaggle setup |
| Deployment | `deploy/huggingface/README.md` | Containerized API, security |
| Architecture decisions | `docs/adr/` | ADRs and design rationale |
| Operations | `docs/runbooks/` | Troubleshooting, monitoring, rollback |
| Validation reports | `docs/validation/` | Benchmark results, monitoring replay |

---

## License

Apache License 2.0

Developed for the CIF / DigiCoop-WA+ ecosystem.
