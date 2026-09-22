# CIF Credit Intelligence — deploiement production (Oracle Cloud Free Tier)
#
# Architecture cible (IaC + GitOps-like) :
#   GitHub Actions  --build/push-->  GHCR (image api:latest, modele cuit)
#   Terraform       --provisionne--> VM A1 + K3s (Oracle Always-Free)
#   K3s             --execute-->     deployment cif-api (2 replicas, NodePort 30080)
#
# Usage (une seule commande) :
#   make deploy \
#     TF_VAR_tenancy_ocid=ocid1.tenancy.. \
#     TF_VAR_user_ocid=ocid1.user.. \
#     TF_VAR_compartment_ocid=ocid1.compartment.. \
#     TF_VAR_fingerprint=xx:xx \
#     TF_VAR_private_key_path=~/.oci/key.pem \
#     TF_VAR_ssh_public_key="ssh-rsa AAAA..." \
#     SSH_KEY=~/.ssh/id_ed25519
#
# Prerequis : terraform, kubectl, scp, curl, openssl installes.
# GHCR : rendre le package `api` public (GitHub -> Packages -> api -> public).

.PHONY: init deploy verify clean

init:
	cd terraform && terraform init -input=false

deploy: init
	./scripts/deploy.sh

verify:
	KUBECONFIG=/tmp/cif-k3s.yaml kubectl -n cif get pods,svc

clean:
	cd terraform && terraform destroy -auto-approve

# --- Données ---
.PHONY: data-download
data-download:
	./scripts/download_lending_club.sh

# --- Pipeline de validation (données publiques Lending Club) ---
.PHONY: ingest benchmark pipeline test lint
ingest:
	cif-ingest-lc

benchmark:
	cif-benchmark

pipeline: ingest benchmark

lint:
	ruff check . && ruff format --check . && mypy src

test:
	pytest -q tests

# --- Verrouillage des dépendances ---
# À relancer après toute modification de pyproject.toml. --python-version 3.11 est
# intentionnel : c'est la version de l'image de production (deploy/huggingface/Dockerfile),
# pas forcément celle de votre environnement local — un vrai incident de version divergente
# (pandas résolu différemment selon l'environnement) a motivé ce verrouillage explicite.
.PHONY: lock
lock:
	pip install -q uv
	uv pip compile pyproject.toml --python-version 3.11 -o requirements-serving.lock.txt
	uv pip compile pyproject.toml --extra dev --extra data --extra repro --extra quality \
		--python-version 3.11 -o requirements-dev.lock.txt
