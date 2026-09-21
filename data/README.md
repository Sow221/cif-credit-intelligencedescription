# Données

Aucune donnée n'est versionnée dans git (`data/raw/*`, `data/processed/*` sont ignorés).
Les données sont reproductibles par script et, à terme, versionnées par DVC.

| Dossier | Contenu | Règle |
|---|---|---|
| `raw/` | Fichiers sources tels que reçus | Immuable : jamais modifié à la main |
| `interim/` | Données nettoyées et validées | Régénérable |
| `processed/` | Tables de features prêtes pour l'entraînement | Régénérable |
| `artifacts/` | Sorties de modèles et rapports | Régénérable |
| `model_cards/`, `monitoring/` | Versionnés : documentation du modèle, baseline de référence | Petits fichiers |

## Jeu Lending Club (données publiques réelles)

Sert à valider la **méthode** (split temporel, baseline, monitoring) sur de vraies données.
Il s'agit de crédit américain entre particuliers : il ne renseigne pas sur la performance
attendue sur le portefeuille CIF.

### Télécharger

1. Créez un compte sur kaggle.com, ouvrez le dataset et acceptez ses conditions d'utilisation.
2. Kaggle → *Settings* → *API* → créer un jeton. Enregistrez `kaggle.json` dans `~/.kaggle/` :
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
   (Sur un format de jeton plus récent, exportez plutôt `KAGGLE_API_TOKEN`.)
3. `pip install -e ".[data]" && make data-download`

Le jeton est un secret : ne le commitez pas et ne le collez dans aucune conversation.
