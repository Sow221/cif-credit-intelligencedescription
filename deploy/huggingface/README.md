# Déploiement — CIF Credit Intelligence

Deux modèles servis par la même API : le **pilote CIF** (synthétique, `/v1/predict`,
25 features) et le **champion Lending Club** (validé sur données publiques réelles,
`/v1/lending-club/score`, 18 features, voir `docs/validation/lending-club-benchmark.md`).
Les deux sont exportés en artefacts autonomes (`deploy/model/`, `deploy/model_lending_club/`)
et cuits dans l'image — aucun serveur MLflow requis en production.

> **Transparence méthodologique** : le modèle CIF est entraîné sur des données
> **synthétiques** (aucune donnée réelle CIF n'a été utilisée) — démontre la
> robustesse du système MLOps, pas un AUC réel sur des clients. Le champion
> Lending Club, lui, est validé sur des données réelles, mais un autre marché
> du crédit (particuliers américains) : il valide la **méthode**, pas la
> performance attendue sur le portefeuille CIF.

## 1. API en local

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
set MLFLOW_TRACKING_URI=sqlite:///mlruns.db
set CIF_ENV=dev
uvicorn api.app:create_app --factory --port 8000
```

En mode `dev` (défaut hors production), un secret JWT aléatoire est généré à chaque
démarrage et le client de démo `cif-agent` / `change-me-in-production` reste accepté.
Token : `POST /v1/auth/token` avec ces identifiants.

## 2. Image Docker (locale)

L'image embarque `CIF_ENV=production` : elle **refuse de démarrer** sans de vrais
secrets (jamais les valeurs par défaut), avec un message d'erreur explicite.

```bash
docker build -f deploy/huggingface/Dockerfile -t cif-api .
docker run -p 8000:8000 \
  -e CIF_JWT_SECRET="$(openssl rand -hex 32)" \
  -e CIF_API_CLIENT_SECRET="$(openssl rand -hex 16)" \
  cif-api
```

Régénérer les artefacts avant de rebuilder (ils sont commités dans `deploy/`, pas
recalculés en CI) :

```bash
MLFLOW_TRACKING_URI=sqlite:///mlruns.db cif-export-model        # pilote CIF
MLFLOW_TRACKING_URI=sqlite:///mlruns.db cif-export-lc-model     # champion Lending Club
```

## 3. CI/CD → GHCR

`push` sur `main` déclenche `.github/workflows/deploy.yml` : build de l'image
(avec le modèle cuit dans `deploy/model/`) et push vers
`ghcr.io/sow221/cif-credit-intelligence/api:latest`.

**À faire une fois** : dans GitHub → Packages → rendre le package `api`
**public** (sinon les runtimes externes ne peuvent pas tirer l'image).

## 4. Hugging Face Space (demo publique)

Créer un Space de type **Docker**, et utiliser `deploy/huggingface/space.Dockerfile`
comme Dockerfile (3 lignes pointant vers l'image GHCR publique). HF expose
automatiquement le port `7860`.

**Obligatoire avant le premier démarrage** : Space → *Settings* → *Variables and secrets*,
ajouter en secrets (pas en variables publiques) :
- `CIF_JWT_SECRET` : `openssl rand -hex 32`
- `CIF_API_CLIENT_SECRET` : `openssl rand -hex 16`

Sans ces deux secrets, le conteneur démarre puis s'arrête immédiatement (`CIF_ENV=production`
refuse les valeurs par défaut) — comportement voulu, pas un bug. `PORT=7860` est déjà fixé par
le `space.Dockerfile`. URL publique obtenue après le build du Space.

## 5. Kubernetes (Oracle Cloud Free Tier)

Le cluster tire l'image GHCR (voir §3 : rendre le package **public**).

**Option A — provisionner l'infra via Terraform** (crée une VM A1 + K3s) :

```bash
cd terraform && terraform init && terraform apply \
  -var="tenancy_ocid=ocid1.tenancy.." \
  -var="user_ocid=ocid1.user.." \
  -var="compartment_ocid=ocid1.compartment.." \
  -var="fingerprint=xx:xx" \
  -var="private_key_path=~/.oci/key.pem" \
  -var="ssh_public_key=ssh-rsa AAAA..."
# cloud-init installe K3s, cree le namespace + le secret JWT, et applique
# deployment.yaml + service.yaml (NodePort 30080).
```

**Option B — cluster deja existant** :

```bash
kubectl apply -f k8s/namespace.yaml -f k8s/secret.yaml \
  -f k8s/deployment.yaml -f k8s/service.yaml
# rotation du secret JWT en prod :
kubectl -n cif create secret generic cif-api-secret \
  --from-literal=jwt=$(openssl rand -hex 32) --dry-run=client -o yaml | kubectl apply -f -
```

Vérification et URL de demo :

```bash
kubectl -n cif get pods
# Service NodePort 30080 -> http://<ip-publique-du-node>:30080/v1/health
```

> `ingress.yaml` est **optionnel** : il nécessite nginx-ingress + cert-manager +
> un ClusterIssuer `cloudflare-origin`. Sans ces composants, exposez via le
> NodePort (ou `kubectl port-forward svc/cif-api 8000:80 -n cif`).

## Sécurité

- `CIF_ENV=production` (déjà fixé dans l'image) : l'API refuse de démarrer si
  `CIF_JWT_SECRET` ou `CIF_API_CLIENT_SECRET` sont absents ou valent leur défaut
  de démo. Pas de faille possible « oubliée » en prod — c'est vérifié au démarrage.
- JWT `HS256`, TTL 1h. CORS ouvert par défaut (`*`) : restreindre en prod.
- Un seul couple identifiant/secret client (`cif-agent`) : suffisant pour une démo,
  pas pour plusieurs clients réels (voir roadmap : gestion multi-clients).
