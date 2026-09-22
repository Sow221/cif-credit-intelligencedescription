# Changelog

Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) ; ce projet suit
[Semantic Versioning](https://semver.org/lang/fr/) une fois publié (voir §Versionnement plus bas).

## [1.0.0] — 2026-09-22

Premier passage en production réelle et publique. Jusqu'ici le dépôt démontrait une
infrastructure MLOps sur un prototype synthétique ; cette version ajoute une méthode validée
sur données réelles, une gouvernance de modèle opérationnelle, et un service effectivement
déployé et vérifié de bout en bout — pas seulement testé en local.

### Ajouté
- **Validation out-of-time sur données publiques réelles** (Lending Club, ~618k prêts,
  2009-2015) : ingestion avec liste blanche de colonnes point-in-time, contrat Pandera bloquant,
  split temporel réel (entraînement ≤2013 / validation 2014 / test 2015, jamais retouché).
  Baseline logistique comparée à un XGBoost contraint (monotonicité, CV temporelle via Optuna) ;
  champion retenu seulement si l'écart d'AUC est démontré statistiquement (IC95% apparié). Voir
  `docs/validation/lending-club-benchmark.md` et les ADR `docs/adr/0001` à `0003`.
- **Gouvernance de modèle par alias MLflow** (`champion`/`previous`, `src/models/promotion.py`) :
  porte de qualité objective (seuils AUC/ECE), rollback en une commande (`cif-rollback`),
  remplace les stages MLflow dépréciés. Réutilisée à l'identique pour les deux modèles servis.
- **Nouvel endpoint `/v1/lending-club/score`** : sert le champion validé, décision à 3 niveaux
  autour d'un seuil optimisé par coût (`evaluation/thresholds.py`, jamais posé à la main),
  explications exactes par variable (contribution linéaire pour le champion logistique,
  contributions XGBoost natives si un futur challenger XGBoost gagne). Distinct et jamais
  mélangé avec `/v1/predict` (pilote CIF synthétique).
- **Orchestration Dagster du pipeline Lending Club** (`lc_raw_available` → `lc_interim` →
  `lc_features` → `lc_benchmark`), visible dans l'UI aux côtés du pipeline synthétique existant.
- **Déploiement public réel et vérifié** : https://cif-credit-intelligence.onrender.com — image
  Docker durcie (utilisateur non privilégié, secrets de production obligatoires, refusés s'ils
  restent à leur valeur par défaut), testée en conditions réelles (build, démarrage, appels HTTP
  contre les deux modèles) avant toute mise en ligne.
- **CI/CD réelle sur GitHub Actions** : scans de sécurité (Trivy sur les images, pip-audit sur
  les dépendances, en rapport non bloquant), cycle d'intégration complet (génération, entraînement,
  MLflow, API, Prometheus/Grafana, Dagster).
- **Verrouillage des dépendances** (`requirements-serving.lock.txt`, `requirements-dev.lock.txt`,
  `make lock`) : versions exactes, résolues pour Python 3.11 (celui de l'image de production).
- `CONTRIBUTING.md`, ce fichier, et une licence cohérente (MIT partout, `pyproject.toml` et
  `LICENSE` étaient contradictoires).

### Modifié
- Séparation des dépendances de service (`dependencies`) et d'entraînement/orchestration
  (extra `training` : Dagster, Optuna, Pandera, Matplotlib) — l'image de production n'installe
  plus d'outillage qu'elle n'utilise jamais.
- `docker/Dockerfile.api` tournait en root ; corrigé (utilisateur non privilégié, comme
  `deploy/huggingface/Dockerfile` l'était déjà).

### Corrigé
Bugs réels, trouvés en exécutant chaque outil pour de vrai plutôt qu'en le supposant fonctionnel :
- Modèle CIF synthétique et images de déploiement pointaient sur `models:/…/latest` au lieu de
  `@champion` dans plusieurs modules (`api/app.py`, `pipelines/definitions.py`, `cli/decision.py`).
- `/metrics` renvoyait du JSON au lieu du format texte Prometheus attendu par un vrai scraper.
- `LendingClubScoreRequest.features` n'acceptait pas `null` pour une valeur numérique inconnue —
  un `NaN` littéral n'est pas du JSON standard, un client HTTP strict (`httpx`) refuse même de
  l'émettre, alors que le modèle sait gérer l'absence d'une variable par imputation.
- `terraform/main.tf` contenait une syntaxe HCL invalide (jamais validée avant cette version) et
  un double provisionnement de la couche applicative créant un secret Kubernetes incomplet.
- Deux versions d'action GitHub (`aquasecurity/trivy-action`) inexistantes ou cassées,
  détectées uniquement en poussant réellement vers GitHub Actions.
- Crash matplotlib (bug connu numpy#10322) sur la génération du graphique de benchmark.

## [0.1.0] — antérieur

Infrastructure MLOps sur prototype synthétique CIF : API FastAPI/JWT, feature engineering avec
garde anti-fuite, entraînement XGBoost calibré, audit trail PostgreSQL, monitoring de dérive
Evidently/Prometheus/Grafana, CI/CD, Terraform/Kubernetes (non déployés à ce stade). Voir
l'historique Git antérieur à ce fichier pour le détail.

## Versionnement

[Semantic Versioning](https://semver.org/lang/fr/) : incrémenter le **majeur** pour un
changement cassant de l'API publique ou de la policy de décision, le **mineur** pour un ajout
rétrocompatible (nouveau modèle, nouvel endpoint), le **correctif** pour un bug corrigé sans
changement de comportement attendu.
