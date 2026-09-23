# Déploiement — CIF Credit Intelligence

> **En production maintenant** : https://cif-credit-intelligence.onrender.com/docs (Render,
> instance gratuite unique — voir « API en production » dans le README racine pour le chemin de
> montée en charge). Ce dossier s'appelle `huggingface/` pour des raisons historiques : Hugging
> Face Spaces a retiré son palier Docker gratuit en 2026 (abonnement payant requis pour créer un
> Space Docker ou Gradio), donc ce chemin n'est plus utilisé. L'image et le `Dockerfile`
> ci-dessous restent corrects et portables — c'est exactement cette image qui tourne sur Render
> aujourd'hui, et elle fonctionnerait identiquement sur Hugging Face si leur politique change.

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

## 4. Service Docker public (production actuelle : Render)

N'importe quelle plateforme qui sait tirer une image publique depuis un registre (GHCR) et lui
injecter des variables d'environnement convient — l'image ne dépend d'aucune spécificité de
plateforme. En production aujourd'hui : **Render**, service Web créé directement depuis
`ghcr.io/sow221/cif-credit-intelligence/api:latest` (pas de build, pas de dépôt Git connecté).

**Obligatoire avant le premier démarrage**, quelle que soit la plateforme — définir en secrets
(jamais en clair, jamais commités) :
- `CIF_JWT_SECRET` : `openssl rand -hex 32`
- `CIF_API_CLIENT_SECRET` : `openssl rand -hex 16`
- `CIF_ENV` : `production`

Sans les deux premiers, le conteneur démarre puis s'arrête immédiatement (`assert_production_secrets`
refuse les valeurs par défaut) — comportement voulu, pas un bug. La plateforme fournit `PORT`
automatiquement (Render : 10000 ; Hugging Face Spaces si réactivé un jour : 7860 via
`space.Dockerfile`) ; l'image le lit dynamiquement, aucune configuration supplémentaire requise.

## 5. Kubernetes (cible de montée en charge, Oracle Cloud Always Free)

Non activée aujourd'hui (production actuelle : §4, Render) — prête à l'être : Terraform
provisionne la VM et installe K3s, `scripts/deploy.sh` applique ensuite les manifestes et crée
les secrets (source unique de vérité pour la couche applicative, jamais dupliquée entre les deux).

**Une seule commande** (voir `Makefile` cible `deploy`) :

```bash
export TF_VAR_tenancy_ocid=ocid1.tenancy..
export TF_VAR_user_ocid=ocid1.user..
export TF_VAR_compartment_ocid=ocid1.compartment..
export TF_VAR_fingerprint=xx:xx
export TF_VAR_private_key_path=~/.oci/key.pem
export TF_VAR_ssh_public_key="ssh-rsa AAAA..."
export SSH_KEY=~/.ssh/id_ed25519
make deploy
```

Ça provisionne la VM (Terraform), installe K3s, récupère le kubeconfig, crée le namespace et un
secret JWT + client généré aléatoirement (jamais de valeur par défaut), applique
`deployment.yaml` (2 réplicas) et `service.yaml`, puis affiche l'URL : `http://<ip>:30080/v1/health`.

**Rotation du secret** sur un cluster déjà déployé :

```bash
KUBECONFIG=/tmp/cif-k3s.yaml kubectl -n cif create secret generic cif-api-secret \
  --from-literal=jwt=$(openssl rand -hex 32) \
  --from-literal=client_secret=$(openssl rand -hex 16) \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n cif rollout restart deployment/cif-api
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

## Rate limiting à plusieurs instances (`CIF_REDIS_URL`)

Le rate limiting par défaut compte en mémoire du processus — correct pour une seule instance
(ce que sert Render aujourd'hui), **incorrect dès qu'il y a plusieurs réplicas** : chaque
processus compte séparément, le quota réel effectif devient `rate_per_minute × nb_instances`.

Définir `CIF_REDIS_URL` (ex. `redis://host:6379`) bascule automatiquement sur un compteur
partagé dans Redis (`src/api/middleware.py::RedisRateLimiter`), vérifié fonctionnel avec deux
conteneurs distincts derrière un même Redis : un quota de 5 partagé, pas 5 par conteneur.
Dégrade proprement si Redis est injoignable au démarrage — l'API revient au comportement
mono-instance plutôt que de refuser de démarrer.

Pertinent pour la cible Kubernetes (`k8s/`, plusieurs réplicas) : ajouter un service Redis au
cluster et définir `CIF_REDIS_URL` dans `k8s/deployment.yaml` avant de monter en charge.
