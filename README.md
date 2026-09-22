# CIF Credit Intelligence

Plateforme de **scoring de crédit** pour l'écosystème CIF / DigiCoop-WA+ — conçue comme une
infrastructure d'intelligence financière (TELQAN Credit), selon les standards d'un cabinet
d'expertise : reproductible, testée, instrumentée, déployable.

> Le ROC-AUC de 0.94 (`cif_credit_official:1.0.1`, voir `data/model_cards/`) du prototype
> synthétique est un **résultat expérimental sur environnement synthétique**, pas un benchmark
> CIF — et il reste structurellement optimiste : le générateur dérive toutes les features d'un
> facteur latent unique, plus séparable qu'un vrai portefeuille de crédit. Ce dépôt contient la
> version *engineering-grade* de ce prototype, prête à accueillir les données réelles CIF via le
> protocole d'audit V1.1.

## API en production

**https://cif-credit-intelligence.onrender.com/docs** — documentation interactive (Swagger),
testable directement dans le navigateur. Sert deux modèles : le pilote CIF synthétique
(`/v1/predict`) et le champion validé sur données publiques réelles Lending Club
(`/v1/lending-club/score`, voir `docs/validation/lending-club-benchmark.md`).

Le déploiement actuel (Render, instance unique à coût nul) est un point de départ délibérément
léger, pas une limite de conception : l'architecture est bâtie pour la montée en charge dès le
premier jour — API sans état (le modèle est chargé une fois, aucune session serveur), registre
de modèles versionné avec promotion/rollback (`src/models/promotion.py`), manifestes Kubernetes
en réplicas (`k8s/deployment.yaml`) et infrastructure-as-code prête (`terraform/`) pour un
cluster dédié quand le trafic le justifiera. Passer à l'échelle est un changement de cible de
déploiement, pas une réécriture.

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
pipelines/                # Définitions Dagster (assets, jobs, schedules)
conf/                     # Configuration Hydra/YAML (data, model, decision, evaluation, serving)
data/                     # Données (non versionnées, voir data/README.md) + model cards + baseline drift
deploy/                   # Artefacts de déploiement : modèles exportés cuits dans l'image, gabarits de service
docker/                   # Dockerfiles (api, dagster, mlflow)
infra/                    # Stack locale : Compose, Prometheus, Grafana, Alertmanager, observability/
k8s/  terraform/          # Cible de montée en charge (K3s + Oracle Cloud), prête, non activée
scripts/                  # Scripts d'exploitation (deploy, téléchargement des données)
tests/                    # Tests unitaires et d'intégration
docs/                     # Charte, ADR (adr/), runbooks (runbooks/), rapports de validation (validation/)
reports/                  # Rapports générés publiables (métriques, backtests)
```

## Stack

| Couche | Outil |
|---|---|
| Langage | Python 3.11, packages typés (mypy strict, ruff) |
| Config | Pydantic v2 (settings) + Hydra (YAML) |
| Orchestration | Dagster (assets versionnés, lineage) |
| Tracking & registry | MLflow (Postgres + MinIO) |
| Qualité de données | Pandera (contrat de données bloquant) + Pydantic v2 |
| Versionnement des données | DVC (`dvc.yaml` : ingest → benchmark) |
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
- **Garde anti-leakage** (`src/features/validate.py` + `src/features/builder.py`) : deux niveaux.
  (1) Blocklist nominale : toute variable de fuite nommée (ex : `p_default_true`) provoque une
  **erreur bloquante** au feature engineering et une **réponse 422** à l'API. Le générateur
  (`src/data/synthetic.py`) **ne diffuse plus** `p_default_true` dans la table clients (purge à la
  source). (2) Garde structurelle (`assert_no_correlation_leakage`) : toute feature dont la
  corrélation avec la cible dépasse 0.75 est bloquante — détecte une fuite **sémantique** qu'un nom
  de colonne innocent ne révélerait pas (cas historique corrigé : `historical_default_rate` dérivée
  d'un `loan_status` lui-même calculé à partir de la cible, corrélation ≈0.94, jamais détectée par
  la seule blocklist nominale). Verrouillé par `tests/unit/test_leakage.py` et
  `tests/integration/test_api.py`.
- **Feature-set = modèle officiel calibré** (`src/features/builder.py`) : les 25 features du `cifci`
  (profil/revenu, agrégation des prêts, ratios dérivés) sont reconstruites comme source de vérité
  unique, avec contrats (`src/features/definitions/`) et tests.
- **Hyperparamètres du modèle alignés sur l'officiel calibré** : `max_depth=4`, `learning_rate=0.03`,
  `n_estimators=300`.

## Validation sur données publiques réelles (Lending Club)

```bash
pip install -e ".[dev,data]"
make data-download      # Kaggle CLI, voir data/README.md
make pipeline           # ingest (validation Pandera) puis benchmark (baseline vs XGBoost)
```

Protocole out-of-time (ADR `docs/adr/`) : split par date d'octroi, baseline logistique, XGBoost
contraint par monotonicité (Optuna, CV temporelle), calibration isotonique sur la validation,
test évalué une fois, champion retenu seulement si l'écart d'AUC est statistiquement démontré.
Ces résultats valident la **méthode**, pas la performance sur le portefeuille CIF.

## État

Phase 1 — Infrastructure (ce dépôt), **terminée et en production**. Phase 2 — Validation de la
méthode sur données publiques réelles (Lending Club), **terminée** — voir « API en production »
ci-dessus et `docs/validation/`. Phase 3 — Protocole données réelles CIF (shadow mode), à venir.

**Rigueur exécutée** : 107 tests verts (unitaires + intégration, tous exécutés en CI), couverture 74 % (seuil CI : 70 %, cible : 90 %), ruff et `mypy --strict` sans erreur, CI et déploiement réels et verts sur GitHub Actions.

## Avancement selon le plan du cabinet (retour.txt)

| Semaine | Livrables | Statut |
|---|---|---|
| 1 — Repo & Fondations | Structure `src/config`, `src/features/definitions/` (25 contrats), settings Pydantic | ✅ `971deeb` |
| 2 — API | `/v1/predict`, `/v1/auth/token`, JWT HS256, rate limiting, `X-Request-ID`, `extra="forbid"` | ✅ `030ece9` |
| 3 — Base de données | `migrations/` (Alembic), `src/services/audit_service.py`, `src/services/predictor.py`, tables PostgreSQL 16 (customers, predictions, audit_log, model_versions) | ✅ |
| 4 — MLflow + Monitoring | `src/models/train.py`, `src/models/model_card.py`, `src/monitoring/drift_report.py` | ✅ `117e8a6` |
| 5 — Kubernetes + Terraform | `terraform/main.tf`, `k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/ingress.yaml`, `docker/Dockerfile.api` | ✅ `637e402` |
| 6 — CI/CD Canary | `.github/workflows/deploy.yml`, `scripts/deploy.sh` | ✅ `61c34e7` — rollback K8s dédié restant à écrire |

Base de données : le schéma (customers, predictions, audit_log, model_versions) est porté par
`migrations/` (Alembic, cible PostgreSQL 16). En local/test, l'audit peut pointer sur SQLite
(`CIF_DATABASE__URL=sqlite+pysqlite:///audit.db`) ; en production le `jsonb` natif PostgreSQL
est utilisé. L'audit est activé automatiquement dès que `CIF_DATABASE__URL` est définie.