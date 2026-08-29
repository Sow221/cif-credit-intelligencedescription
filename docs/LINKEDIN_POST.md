# Post LinkedIn — CIF Credit Intelligence (à publier après déploiement)

---

🚀 J'ai conçu et déployé une plateforme complète de scoring de crédit MLOps —
du notebook à la production.

Contexte : la décision de crédit en zone UEMOA repose encore largement sur des
grilles manuelles. J'ai construit un système de scoring crédit end-to-end,
méthodologiquement rigoureux :

✅ 25 features officielles du modèle de référence CIF, reconstruites proprement
   (pas de fuite de données : split temporel + garde anti-leakage bloquante par défaut)
✅ Pipeline de feature engineering 100% déterministe
   (savings rolling 24m, historique de crédit agrégé, ratios)
✅ Modèle XGBoost calibré (Platt) avec Brier + ECE mesurés (calibration suivie)
✅ API de scoring en production (FastAPI + JWT, ruff + mypy --strict, 72 tests)
✅ Déploiement : Docker, Kubernetes (Oracle Cloud Free Tier), Terraform,
   CI/CD GitHub Actions → image GHCR, demo Hugging Face Spaces

🔎 Détail de transparence (comme un vrai data engineer le documenterait) : les
données d'entraînement sont **synthétiques** — je n'ai jamais eu accès aux
données réelles CIF. L'objectif de cette preuve est de démontrer la
**robustesse du système MLOps** (reproductible, versionné, déployable, honnête
sur ses limites), pas de prétendre à un AUC réel sur des clients. C'est
exactement ce qu'on attend d'un senior : un pipeline qui tient la route, pas
un chiffre惊人 sorti d'un notebook.

Tech : Python · XGBoost · MLflow · FastAPI · Docker · Kubernetes · Terraform · GitHub Actions

🧠 Code + infra : https://github.com/Sow221/cif-credit-intelligencedescription
🌐 Démo live : [URL_HF_SPACES_A_REMPLIR]

#MachineLearning #CreditScoring #MLOps #DataEngineering #FinTech #Python #Kubernetes #Terraform

---

Notes d'édition :
- Remplacer [URL_HF_SPACES_A_REMPLIR] par l'URL du Space une fois déployé.
- Ne pas présenter le ROC-AUC (0.994) comme "réel" : préciser "sur données synthétiques".
