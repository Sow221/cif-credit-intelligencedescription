#!/usr/bin/env bash
# Télécharge le jeu Lending Club (Kaggle) dans data/raw/lending_club/ — reproductible, jamais commité.
#
# Prérequis :
#   pip install -e ".[data]"
#   Un jeton API Kaggle dans ~/.kaggle/kaggle.json (chmod 600) OU la variable d'environnement
#   KAGGLE_API_TOKEN. Voir data/README.md. Ne jamais coller ce jeton dans le dépôt ni dans un chat.
#   Accepter les conditions du dataset sur sa page Kaggle avant le premier téléchargement.
set -euo pipefail

DATASET="${LENDING_CLUB_DATASET:-wordsforthewise/lending-club}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw/lending_club"

command -v kaggle >/dev/null || { echo "kaggle CLI absente : pip install -e '.[data]'" >&2; exit 1; }
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -z "${KAGGLE_KEY:-}" ]; then
  echo "Identifiants Kaggle introuvables (voir data/README.md)." >&2
  exit 1
fi

mkdir -p "$DEST"
echo ">>> Téléchargement de $DATASET vers $DEST"
kaggle datasets download -d "$DATASET" -p "$DEST" --unzip
echo ">>> Contenu :"
ls -lh "$DEST"
