# CIF Credit Intelligence

Plateforme de **scoring de crédit** pour l'écosystème CIF / DigiCoop-WA+ — conçue comme une
infrastructure d'intelligence financière (TELQAN Credit), selon les standards d'un cabinet
d'expertise : reproductible, testée, instrumentée, déployable.

> Le ROC-AUC de 0.83 du prototype synthétique est un **résultat expérimental sur environnement
> synthétique**, pas un benchmark CIF. Ce dépôt contient la version *engineering-grade* de ce
> prototype, prête à accueillir les données réelles CIF via le protocole d'audit V1.1.

## Architecture (alignée sur le cahier du cabinet)

```
src/
├── api/                  # API FastAPI : app.py, routes.py, schemas.py, middleware.py
├── config/               # settings.py (Pydantic) + schema (Hydra) + load
├── data/                 # Générateur synthétique
├── evaluation/           # Métriques, calibration, bootstrap, robustesse, fairness
├── features/
│   ├── definitions/      # UN fichier par feature + contrat (nom, version, bornes, owner, SLA)
│   └── builder.py        # Calcul centralisé (aucune formule dupliquée)
├── models/               # Entraînement XGBoost (train.py) + model_card.py
├── monitoring/           # Drift (Evidently) : drift_report.py, metrics Prometheus
├── services/             # decision_engine.py, confidence.py, predictor.py, audit_service.py
├── cli/                  # Commandes cif-*
└── utils/                # Logging structuré, reproductibilité
migrations/               # Alembic (schema PostgreSQL versionné)
terraform/                # Infrastructure as Code (Oracle Cloud)
k8s/                      # Manifests Kubernetes (deployment, service, ingress, rollback)
pipelines/                # Définitions Dagster (assets, jobs, schedules)
infra/                    # Docker Compose, Grafana, Prometheus, MLflow
tests/                    # Tests unitaires et d'intégration
docs/                     # ADR, model cards, runbooks
```

## Stack

| Couche | Outil |
|---|---|
| Langage | Python 3.11, packages typés (mypy strict, ruff) |
| Config | Pydantic v2 (settings) + Hydra (YAML) |
| Orchestration | Dagster (assets versionnés, lineage) |
| Tracking & registry | MLflow (Postgres + MinIO) |
| Qualité de données | Great Expectations + Pydantic v2 |
| Monitoring | Evidently + Prometheus + Grafana + Alertmanager |
| Serving | FastAPI, modèle servi depuis le registry |
| BDD | PostgreSQL 16 + Alembic (audit trail) |
| CI/CD | GitHub Actions (lint → mypy → tests → build → promote) |
| Déploiement | K3s sur Oracle Cloud Free Tier + Terraform |

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
séparation Model/Policy/Workflow, human-in-the-loop, gouvernance des modèles, monitoring et rollback.
Voir `docs/` pour le détail et `traçabilité.md` du prototype.

### Validation méthodologique (fusion avec le dépôt `cifci`)

Les garanties de validité du protocole CIF sont **implémentées dans le code**, pas seulement
documentées :

- **Split temporel** (`src/models/train.py` → `temporal_split`) : l'entraînement n'utilise **jamais**
  de split aléatoire. La coupure suit l'ordre temporel (proxy `customer_id` sur le jeu synthétique,
  à substituer par une vraie colonne `application_date` sur les données réelles). Verrouillé par
  `tests/unit/test_temporal_split.py`.
- **Garde anti-leakage** (`src/features/validate.py` + `src/features/builder.py`) : toute variable de
  fuite (ex : `p_default_true`) provoque une **erreur bloquante** au feature engineering et une
  **réponse 422** à l'API. Le générateur (`src/data/synthetic.py`) **ne diffuse plus** `p_default_true`
  dans la table clients (purge à la source). Verrouillé par `tests/unit/test_leakage.py` et
  `tests/integration/test_api.py`.
- **Hyperparamètres du modèle alignés sur l'officiel calibré** : `max_depth=4`, `learning_rate=0.03`,
  `n_estimators=300`.

## État

Phase 1 — Infrastructure (ce dépôt). Phase 2 — Reproduction du prototype synthétique.
Phase 3 — Protocole données réelles CIF (shadow mode).

**Rigueur exécutée** : 71 tests verts (52 unitaires + 16 intégration + 3 temporel), ruff et
`mypy --strict` sans erreur.

## Avancement selon le plan du cabinet (retour.txt)

| Semaine | Livrables | Statut |
|---|---|---|
| 1 — Repo & Fondations | Structure `src/config`, `src/features/definitions/` (25 contrats), settings Pydantic | ✅ `971deeb` |
| 2 — API | `/v1/predict`, `/v1/auth/token`, JWT HS256, rate limiting, `X-Request-ID`, `extra="forbid"` | ✅ `030ece9` |
| 3 — Base de données | `migrations/` (Alembic), `src/services/audit_service.py`, `src/services/predictor.py`, tables PostgreSQL 16 (customers, predictions, audit_log, model_versions) | ✅ |
| 4 — MLflow + Monitoring | `src/models/train.py`, `src/models/model_card.py`, `src/monitoring/drift_report.py` | ⏳ |
| 5 — Kubernetes + Terraform | `terraform/main.tf`, `k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/ingress.yaml`, `docker/Dockerfile` | ⏳ |
| 6 — CI/CD Canary | `.github/workflows/deploy.yml`, `k8s/rollback.yaml` | ⏳ |

Base de données : le schéma (customers, predictions, audit_log, model_versions) est porté par
`migrations/` (Alembic, cible PostgreSQL 16). En local/test, l'audit peut pointer sur SQLite
(`CIF_DATABASE__URL=sqlite+pysqlite:///audit.db`) ; en production le `jsonb` natif PostgreSQL
est utilisé. L'audit est activé automatiquement dès que `CIF_DATABASE__URL` est définie.