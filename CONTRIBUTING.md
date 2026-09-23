# Contribuer à CIF Credit Intelligence

Merci de vouloir contribuer. Ce document explique comment installer le projet, ce qui est
attendu d'une Pull Request, et — le plus important — **comment proposer un modèle candidat**,
parce que c'est le cas d'usage le plus probable pour ce dépôt.

## Installation

```bash
git clone <ce dépôt>
cd cif-credit-intelligencedescription
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data,repro]"
```

`requirements-dev.lock.txt` fixe les versions exactes utilisées en CI (généré pour Python 3.11
via `make lock`) ; `pip install -e ".[dev]"` résout librement et peut donc dériver dans le
temps — en cas de comportement différent de la CI, comparez d'abord avec le fichier verrouillé.

```bash
make lint    # ruff + mypy --strict
make test    # pytest (unitaires + intégration)
```

Les deux doivent passer avant toute Pull Request — la CI les relance de toute façon sur GitHub
Actions (`.github/workflows/ci.yml`) dès l'ouverture, pas seulement sur `main`.

## Style et conventions

- Python 3.11+, typage strict (`mypy --strict` sans erreur, pas de `# type: ignore` sans commentaire).
- `ruff format` avant de committer (`make lint` te le signale, ne le fait pas à ta place).
- Docstrings en français, ce que fait le reste du dépôt — explique le *pourquoi*, pas seulement
  le *quoi* (le code dit déjà le quoi).
- Un commit = un changement cohérent, message qui explique la raison, pas juste la liste des
  fichiers touchés.

## Proposer un modèle candidat (le cas d'usage principal de ce dépôt)

Le projet n'accepte pas un nouveau modèle « parce qu'il a un meilleur score ». Le protocole
est fixé à l'avance et documenté (`docs/adr/`), pas négociable au cas par cas :

1. **Split temporel réel**, jamais aléatoire. Si tes données ont une colonne de date, utilise-la ;
   sinon documente explicitement le proxy choisi et sa limite (voir `docs/adr/0001` pour un exemple).
2. **Compare à la baseline existante**, pas seulement au champion actuel. Une régression
   logistique reste la référence par défaut du domaine — voir `src/models/scoring.py::fit_logistic`.
3. **La promotion n'est jamais automatique sur un chiffre brut.** `src/models/promotion.py`
   n'accepte un candidat que si l'écart d'AUC avec le champion actuel est démontré par un
   intervalle de confiance bootstrap **apparié** qui exclut 0 (voir `evaluation/bootstrap.py::paired_auc_difference`
   et `docs/adr/0003`). Un score plus élevé sans cette preuve statistique ne suffit pas.
4. **Aucune variable d'après-décision.** Le contrat de données (`src/data/validation.py` pour
   Lending Club, `src/features/validate.py` pour le pilote CIF) rejette toute colonne connue
   seulement après l'octroi. Une feature qui améliore le score sans explication causale
   plausible mérite d'être suspectée avant d'être célébrée.
5. **Explique tes seuils de décision par un coût**, pas à la main
   (`evaluation/thresholds.py::optimal_cost_threshold`).
6. Documente la décision dans un nouvel ADR (`docs/adr/000N-*.md`) si elle change une règle du
   protocole — pas seulement dans la description de la Pull Request, qui se perd avec le temps.

Le pipeline complet (`make ingest && make benchmark`, ou l'équivalent Dagster,
`pipelines/definitions.py`) doit passer avant de proposer une promotion.

## Rapporter un bug

Un rapport utile inclut : la commande exacte lancée, ce qui était attendu, ce qui s'est passé,
et si possible la sortie de `pip freeze` ou une référence au fichier de verrouillage utilisé —
plusieurs bugs réels de ce dépôt (voir `CHANGELOG.md`) venaient d'une divergence de version de
dépendance invisible sans cette information.

## Ce qui n'est volontairement pas encore fait

Avant de proposer une contribution qui en dépend, vérifie l'état réel dans `CHANGELOG.md` et les
issues ouvertes : couverture de tests à 90 % (87 % aujourd'hui, seuil CI à 70 %), Redis non
encore provisionné sur la cible Kubernetes (le code le
supporte via `CIF_REDIS_URL`, voir `deploy/huggingface/README.md`, mais rien ne le déploie encore).
