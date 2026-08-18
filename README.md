# CIF Credit Intelligence

Plateforme de **scoring de crédit** pour l'écosystème CIF / DigiCoop-WA+ — conçue comme une
infrastructure d'intelligence financière (TELQAN Credit), selon les standards d'un cabinet
d'expertise : reproductible, testée, instrumentée, déployable.

> Le ROC-AUC de 0.83 du prototype synthétique est un **résultat expérimental sur environnement
> synthétique**, pas un benchmark CIF. Ce dépôt contient la version *engineering-grade* de ce
> prototype, prête à accueillir les données réelles CIF via le protocole d'audit V1.1.

## Architecture

```
src/cif_credit/
├── config/        # Dataclasses + chargement Hydra
├── data/          # Générateur synthétique, schémas, qualité (Great Expectations)
├── features/      # Feature engineering versionné
├── models/        # Entraînement XGBoost, registry MLflow
├── evaluation/    # Métriques, calibration, bootstrap, robustesse, fairness
├── decision/      # Decision engine (Model ≠ Policy ≠ Workflow ≠ Decision)
├── monitoring/    # Dérive (Evidently), métriques Prometheus
├── serving/       # API de scoring FastAPI
└── utils/         # Logging structuré, observabilité (OpenTelemetry)
pipelines/         # Définitions Dagster (assets, jobs, schedules, sensors)
conf/              # Configuration Hydra versionnée
infra/             # Docker Compose, Grafana, Prometheus, MLflow
tests/             # Tests unitaires et d'intégration
docs/              # ADR, model cards, runbooks
```

## Stack

| Couche | Outil |
|---|---|
| Langage | Python 3.11, package `src/` typé (mypy strict, ruff) |
| Config | Hydra (framework de Meta) |
| Orchestration | Dagster (assets versionnés, lineage) |
| Tracking & registry | MLflow (Postgres + MinIO) |
| Qualité de données | Great Expectations + Pydantic v2 |
| Monitoring | Evidently + Prometheus + Grafana + Alertmanager |
| Serving | FastAPI, modèle servi depuis le registry |
| CI/CD | GitHub Actions (lint → mypy → tests → build → promote) |

## Démarrage rapide

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[dev]"

# Générer les données synthétiques
cif-generate
# Entraîner (tracking MLflow local)
cif-train
# Évaluer
cif-evaluate
```

### Stack complète (Docker)

```bash
docker compose -f infra/docker-compose.yml up -d
# MLflow      → http://localhost:5000
# Dagster     → http://localhost:3000
# Grafana     → http://localhost:3001
# API scoring → http://localhost:8000/docs
```

## Conformité cahier des charges CIF

Ce dépôt implémente les exigences §M07 / §64 / §65 / §93 du cahier des charges CIF Digital Platform :
séparation Model/Policy/Workflow, human-in-the-loop, gouvernance des modèles, split temporel,
monitoring et rollback. Voir `docs/` pour le détail.

## État

Phase 1 — Infrastructure (ce dépôt). Phase 2 — Reproduction du prototype synthétique.
Phase 3 — Protocole données réelles CIF (shadow mode).
