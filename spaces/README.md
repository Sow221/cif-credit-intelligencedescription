# Déploiement — CIF Credit Intelligence

Système de scoring de crédit MLOps complet : feature engineering déterministe
(25 features officielles CIF, anti-leakage), modèle XGBoost calibré, API de
production (FastAPI + JWT), et stack de déploiement conteneurisé
(Docker, Kubernetes, Terraform, CI/CD).

> **Transparence méthodologique** : le modèle est entraîné sur des données
> **synthétiques** (aucune donnée réelle CIF n'a été utilisée). L'objectif de
> cette preuve est de démontrer la **robustesse du système MLOps** (pipeline
> reproductible, versionné, déployable, honnête sur ses limites), et non un
> AUC réel sur des clients.

## 1. API en local

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
set MLFLOW_TRACKING_URI=sqlite:///mlruns.db
uvicorn api.app:create_app --factory --port 8000
```

Token : `POST /v1/auth/token` avec `client_id=cif-agent`,
`client_secret=change-me-in-production`.

## 2. Image Docker (locale)

```bash
docker build -f spaces/Dockerfile -t cif-api .
docker run -p 8000:8000 cif-api
```

## 3. CI/CD → GHCR

`push` sur `main` déclenche `.github/workflows/deploy.yml` : build de l'image
(avec le modèle cuit dans `model_export/`) et push vers
`ghcr.io/sow221/cif-credit-intelligence/api:latest`.

**À faire une fois** : dans GitHub → Packages → rendre le package `api`
**public** (sinon les runtimes externes ne peuvent pas tirer l'image).

## 4. Hugging Face Space (demo publique)

Créer un Space de type **Docker**, et utiliser `spaces/space.Dockerfile`
comme Dockerfile (3 lignes pointant vers l'image GHCR publique). HF expose
automatiquement le port `7860`. URL publique obtenue après le build du Space.

## 5. Kubernetes (Oracle Cloud Free Tier)

```bash
cd terraform && terraform init && terraform apply   # VM A1 + K3s
scp -i key.pem k8s/*.yaml ubuntu@<ip>:/tmp/
kubectl apply -f k8s/secret.yaml           # renseigner client_id/secret
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/ingress.yaml
```

## Sécurité

- `cif-agent` / `change-me-in-production` : **à remplacer** par des secrets
  réels (env / secret K8s) avant toute exposition publique.
- JWT `HS256`, TTL 1h. CORS ouvert par défaut (`*`) : restreindre en prod.
