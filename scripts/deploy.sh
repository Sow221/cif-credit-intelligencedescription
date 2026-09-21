#!/usr/bin/env bash
# Deploiement automatise CIF : Terraform (VM K3s OCI) + apply K8s + healthcheck.
set -euo pipefail
cd "$(dirname "$0")/.."

require() { : "${!1:?Variable $1 requise (exportez-la avant 'make deploy')}"; }
require TF_VAR_tenancy_ocid
require TF_VAR_user_ocid
require TF_VAR_compartment_ocid
require TF_VAR_fingerprint
require TF_VAR_private_key_path
require TF_VAR_ssh_public_key
require SSH_KEY

cd terraform
terraform init -input=false
terraform apply -auto-approve

IP=$(terraform output -raw instance_public_ip)
echo ">>> Instance K3s disponible sur $IP"

# Recuperation du kubeconfig depuis le node K3s
KCFG=/tmp/cif-k3s.yaml
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" "ubuntu@$IP:/etc/rancher/k3s/k3s.yaml" "$KCFG"
# Le kubeconfig pointe sur 127.0.0.1 : on remplace par l'IP publique
case "${OSTYPE:-}" in
  darwin*) sed -i '' "s/127.0.0.1/$IP/" "$KCFG" ;;
  *)       sed -i  "s/127.0.0.1/$IP/" "$KCFG" ;;
esac
export KUBECONFIG="$KCFG"

# Secrets générés à la volée : jamais de valeur par défaut committée (cf. k8s/secret.yaml).
kubectl apply -f ../k8s/namespace.yaml
kubectl -n cif create secret generic cif-api-secret \
  --from-literal=jwt="$(openssl rand -hex 32)" \
  --from-literal=client_secret="$(openssl rand -hex 16)" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f ../k8s/deployment.yaml -f ../k8s/service.yaml
kubectl -n cif rollout status deployment/cif-api --timeout=180s

echo
echo "==================================================="
echo " DEMO EN LIGNE : http://$IP:30080/v1/health"
echo "==================================================="
