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
