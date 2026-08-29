# Post LinkedIn — CIF Credit Intelligence (prêt à publier)

---

🚀 J'ai conçu, codé et déployé une plateforme de scoring de crédit MLOps de bout en bout — du notebook à la production sur Oracle Cloud.

Le contexte : en zone UEMOA, la décision de crédit reste largement portée par des grilles manuelles. J'ai construit un système de scoring crédit **méthodologiquement irréprochable et industrialisable** :

🔧 Feature engineering déterministe — les 25 features officielles du modèle de référence CIF, reconstruites proprement avec une **garde anti-leakage bloquante par défaut** (pas de fuite de données : split temporel + validation).
🤖 Modèle XGBoost **calibré** (Platt) — Brier et ECE suivis (on mesure la calibration, pas seulement l'AUC).
🛡️ API de production — FastAPI + JWT, code strict (ruff + mypy --strict), **72 tests verts**, sécurité non-root en conteneur.
⚙️ Infrastructure as Code — **Terraform** provisionne un cluster K3s (Oracle Always-Free), **CI/CD GitHub Actions** build et pousse l'image sur GHCR, **Kubernetes** orchestre le service (2 réplicas, healthchecks, secrets externalisés).
📦 Déploiement one-command : `make deploy`.

🔎 Transparence (ce qui sépare un senior d'un notebook magique) : le modèle est entraîné sur des données **synthétiques** — je n'ai jamais eu accès aux données réelles CIF. Cette preuve démontre la **robustesse du système MLOps** (reproductible, versionné, déployable, honnête sur ses limites), pas un AUC "client". Un vrai data/ML engineer livre un pipeline qui tient la route, pas un chiffre sorti de nulle part.

Stack : Python · XGBoost · MLflow · FastAPI · Docker · Kubernetes · Terraform · GitHub Actions · Oracle Cloud

🧠 Code + infra (IaC + CI/CD complets) : https://github.com/Sow221/cif-credit-intelligencedescription
🌐 Démo en ligne (Oracle K8s) : [URL_A_RENSEIGNER_APRES_DEPLOIEMENT]

#MachineLearning #CreditScoring #MLOps #DataEngineering #FinTech #Python #Kubernetes #Terraform #OracleCloud

---

Notes d'édition :
- Remplacer [URL_A_RENSEIGNER_APRES_DEPLOIEMENT] par l'IP:30080 après `make deploy`.
- Ne jamais présenter le ROC-AUC (0.994) comme "réel" : préciser "sur données synthétiques".
