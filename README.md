# CIF Credit Intelligence

| Élément | Détail |
|---|---|
| Type | Plateforme de scoring de crédit |
| Domaine | CIF / DigiCoop-WA+ |
| Langage principal | Python 3.11 |
| Licence | Apache 2.0 |
| API de production | https://cif-credit-intelligence.onrender.com/docs |
| Statut | Production |

Plateforme de scoring de crédit pour l'écosystème CIF / DigiCoop-WA+, conçue comme une infrastructure d'intelligence financière selon les standards d'un cabinet d'expertise : reproductible, testée, instrumentée, déployable.

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [API en production](#api-en-production)
- [Architecture](#architecture)
- [Stack technique](#stack-technique)
- [Démarrage rapide](#démarrage-rapide)
- [Conformité CIF](#conformité-cif)
- [Validation](#validation)
- [État du projet](#état-du-projet)
- [Documentation complémentaire](#documentation-complémentaire)

## Vue d'ensemble

Le ROC-AUC de 0.94 (`cif_credit_official:1.0.1`) du prototype synthétique est un résultat expérimental sur environnement synthétique, pas un benchmark CIF. Il reste structurellement optimiste : le générateur dérive toutes les features à partir d'un facteur latent unique, plus séparable qu'un vrai portefeuille de crédit.

Ce dépôt contient la version engineering-grade de ce prototype, prête à accueillir les données réelles CIF via le protocole d'audit V1.1.

## API en production

https://cif-credit-intelligence.onrender.com/docs

Documentation interactive (Swagger), testable directement dans le navigateur. La plateforme sert deux modèles :

- le pilote CIF synthétique via `/v1/predict`
- le champion validé sur données publiques réelles Lending Club via `/v1/lending-club/score`

Voir `docs/validation/lending-club-benchmark.md` pour le détail de la validation du benchmark public.

### Infrastructure de déploiement

Le déploiement actuel, basé sur Render avec une instance unique à coût nul, est un point de départ délibérément léger, pas une limite de conception. L'architecture est construite pour la montée en charge dès le premier jour :

- API sans état ; le modèle est chargé une fois et aucune session serveur n'est maintenue
- registre de modèles versionné avec promotion et rollback (`src/models/promotion.py`)
- manifestes Kubernetes en réplicas (`k8s/deployment.yaml`)
- infrastructure as code prête (`terraform/`) pour un cluster dédié dès que le trafic le justifie

Passer à l'échelle est un changement de cible de déploiement, pas une réécriture.

## Architecture

Structure du code alignée sur le cahier du cabinet :

```text
src/
├── api/                  # API FastAPI : app.py, routes.py, schemas.py, middleware.py
├── config/               # settings.py (Pydantic) + schema (Hydra) + load
├── data/                 # Générateur synthétique
├── evaluation/           # Métriques, calibration, bootstrap, robustesse, fairness
├── features/
│   ├── definitions/      # Un fichier par feature + contrat (nom, version, bornes, owner, SLA)
│   └── builder.py        # Calcul centralisé (aucune formule dupliquée)
├── models/               # Entraînement XGBoost (train.py) + model_card.py
├── monitoring/           # Drift (Evidently) : drift_report.py, metrics Prometheus
├── services/             # decision_engine.py, confidence.py, predictor.py, audit_service.py
├── cli/                  # Commandes cif-*
├── utils/                # Logging structuré, reproductibilité
├── config/               # Paramétrage applicatif partagé
├── schemas/              # Contrats externes et internes
├── validation/           # Vérifications de qualité et de conformité
├── telemetry/            # Observabilité logique et opérationnelle
├── cli/                  # Commandes utilisateur et déploiement
├── api/                  # Entrées/sorties de service et routes
├── jobs/                 # Jobs workers, ETL, orchestre
├── workflows/            # Pipelines et orchestration
└── utils/                # Helpers système et sécurité

migrations/               # Alembic (schema PostgreSQL versionné)
pipelines/                # Définitions Dagster (assets, jobs, schedules)
conf/                     # Configuration Hydra/YAML (data, model, decision, evaluation, serving)
data/                     # Données non versionnées + model cards + baseline drift
deploy/                   # Artefacts de déploiement : modèles exportés cuits dans l'image
docker/                   # Dockerfiles (api, dagster, mlflow)
infra/                    # Stack locale : Compose, Prometheus, Grafana, Alertmanager, observability/
k8s/ terraform/           # Cible de montée en charge (K3s + Oracle Cloud), prête, non activée
scripts/                  # Scripts d'exploitation (deploy, téléchargement des données)
tests/                    # Tests unitaires et d'intégration
docs/                     # Charte, ADR (adr/), runbooks (runbooks/), rapports de validation (validation/)
reports/                  # Rapports générés publiables (métriques, backtests)
```

## Stack technique

| Couche | Outil |
|---|---|
| Langage | Python 3.11, packages typés (mypy strict, ruff) |
| Configuration | Pydantic v2 (settings) + Hydra (YAML) |
| Orchestration | Dagster (assets versionnés, lineage) |
| Tracking & registry | MLflow (Postgres + MinIO) |
| Qualité des données | Pandera (contrat bloquant) + Pydantic v2 |
| Versionnement des données | DVC (`dvc.yaml` : ingest → benchmark) |
| Monitoring | Evidently + Prometheus + Grafana + Alertmanager |
| Serving | FastAPI, modèle servi depuis le registry |
| Base de données | PostgreSQL 16 + Alembic (audit trail) |
| CI/CD | GitHub Actions (lint → mypy → tests → build → promote) |
| Déploiement | Render (instance unique) — K3s + Terraform (Oracle Cloud) prêts, non activés |

## Démarrage rapide

### Installation locale

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Unix/macOS
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
```

Services disponibles :

- MLflow : http://localhost:5000
- Dagster : http://localhost:3000
- Grafana : http://localhost:3001
- API scoring : http://localhost:8000/docs

## Conformité CIF

Ce dépôt implémente les exigences §M07 / §64 / §65 / §93 du cahier des charges CIF Digital Platform :

- séparation Model / Policy / Workflow
- human-in-the-loop
- gouvernance des modèles
- monitoring et rollback

Voir `docs/` pour le détail et `traçabilité.md` du prototype.

### Validation méthodologique

Les garanties de validité du protocole CIF sont implémentées dans le code, pas seulement documentées.

#### Split temporel

`src/models/train.py` → `temporal_split`

L'entraînement n'utilise jamais de split aléatoire. La coupure suit l'ordre temporel (proxy `customer_id` sur le jeu synthétique, à substituer par une vraie colonne `application_date` sur les données réelles). Cette règle est verrouillée par `tests/unit/test_temporal_split.py`.

#### Garde anti-leakage

`src/features/validate.py` + `src/features/builder.py` : deux niveaux.

1. Blocklist nominale : toute variable de fuite nommée, par exemple `p_default_true`, provoque une erreur bloquante au feature engineering et une réponse 422 à l'API. Le générateur (`src/data/synthetic.py`) ne diffuse plus `p_default_true` dans la table clients.
2. Garde structurelle : `assert_no_correlation_leakage` ; toute feature dont la corrélation avec la cible dépasse 0.75 est bloquante. Cela détecte une fuite sémantique qu'un nom de colonne innocent ne révélerait pas. Un cas historique corrigé est celui de `historical_default_rate`, dérivée d'un `loan_status` calculé à partir de la cible. La corrélation atteignait environ 0.94, et elle n'aurait pas été détectée par la seule blocklist nominale.

Cette protection est verrouillée par `tests/unit/test_leakage.py` et `tests/integration/test_api.py`.

#### Feature-set = modèle officiel calibré

`src/features/builder.py` : les 25 features du `cifci` (profil/revenu, agrégation des prêts, ratios dérivés) sont reconstruites comme source de vérité unique, avec contrats (`src/features/definitions/`) et tests.

#### Hyperparamètres du modèle

Les hyperparamètres sont alignés sur la version officiellement calibrée :

- `max_depth=4`
- `learning_rate=0.03`
- `n_estimators=300`

## Validation

### Données publiques réelles (Lending Club)

```bash
pip install -e ".[dev,data]"
make data-download      # Kaggle CLI, voir data/README.md
make pipeline           # ingest (validation Pandera) puis benchmark (baseline vs XGBoost)
cif-replay-monitoring   # rejeu de monitoring sur historique (dérive + performance retardée)
```

Le protocole out-of-time, décrit dans les ADR et rapports de validation, repose sur :

- split par date d'octroi
- baseline logistique
- XGBoost contraint par monotonicité (Optuna, CV temporelle)
- calibration isotonique sur la validation
- test évalué une fois ; le champion est retenu seulement si l'écart d'AUC est statistiquement démontré

Ces résultats valident la méthode, pas la performance sur le portefeuille CIF.

Le rejeu de monitoring (`docs/validation/monitoring-replay.md`) simule la surveillance de production sur 24 mois d'historique réel : la dérive (PSI) est disponible immédiatement, tandis que la performance réelle n'est disponible que sur les 12 mois de test, jamais touchés à l'entraînement/calibration et avec le décalage de 36 mois qu'aurait connu un vrai déploiement. Résultat non provoqué : 4 mois déclenchent une alerte PSI sur une même variable, et la performance reste stable sur toute la période.

## État du projet

| Phase | Objectif | Statut | Détail |
|---|---|---|---|
| 1 | Infrastructure | Terminée et en production | Ce dépôt |
| 2 | Validation sur données publiques réelles | Terminée | Voir « API en production » et `docs/validation/` |
| 3 | Protocole données réelles CIF (shadow mode) | À venir | - |

### Rigueur exécutée

- 143 tests verts (unitaires + intégration, tous exécutés en CI)
- couverture estimée à 87 % (seuil CI : 80 %)
- couverture des 11 commandes CLI via `click.testing.CliRunner`

### Avancement selon le plan du cabinet

| Semaine | Livrables | Statut |
|---|---|---|
| 1 — Repo & Fondations | Structure `src/config`, `src/features/definitions/` (25 contrats), settings Pydantic | ✅ `971deeb` |
| 2 — API | `/v1/predict`, `/v1/auth/token`, JWT HS256, rate limiting, `X-Request-ID`, `extra="forbid"` | ✅ `030ece9` |
| 3 — Base de données | `migrations/` (Alembic), `src/services/audit_service.py`, `src/services/predictor.py`, tables PostgreSQL 16 (customers, predictions, audit_log, model_versions) | ✅ |
| 4 — MLflow + Monitoring | `src/models/train.py`, `src/models/model_card.py`, `src/monitoring/drift_report.py` | ✅ `117e8a6` |
| 5 — Kubernetes + Terraform | `terraform/main.tf`, `k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/ingress.yaml`, `docker/Dockerfile.api` | ✅ `637e402` — manifestes écrits, non déployés |
| 6 — CI/CD Canary | `.github/workflows/deploy.yml`, `scripts/deploy.sh` | ✅ `61c34e7` — rollback K8s dédié restant à écrire |

### Base de données

Le schéma (customers, predictions, audit_log, model_versions) est porté par `migrations/` (Alembic, cible PostgreSQL 16). En local et en test, l'audit peut pointer sur SQLite (`CIF_DATABASE__URL=sqlite+pysqlite:///audit.db`). En production, le `jsonb` natif PostgreSQL est utilisé. L'audit est activé automatiquement dès que `CIF_DATABASE__URL` est défini.

## Documentation complémentaire

- `data/README.md` — gouvernance et téléchargement des jeux de données
- `deploy/huggingface/README.md` — déploiement conteneurisé et sécurité
- `docs/adr/` — décisions architecturales
- `docs/runbooks/` — procédures de pilotage et support
- `docs/validation/` — rapports de benchmark et monitoring

---

Dépôt Apache 2.0 — CIF / DigiCoop-WA+
