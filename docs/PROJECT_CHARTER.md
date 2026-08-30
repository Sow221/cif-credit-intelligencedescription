# PROJECT CHARTER — CIF Credit Intelligence

> Statut : **v1.0 — document de référence** · Domaine : scoring de crédit (Afrique de l'Ouest / UEMOA)
> Cadre de travail : TDSP (Microsoft) + pratiques MLOps niveau 1 (Google) · Droits : open source, driver par objectifs.

---

## 1. Mission

Concevoir, construire et **déployer une plateforme de scoring de crédit bout-en-bout** — du pipeline de
données au modèle, de l'API à l'observabilité — avec un niveau de rigueur industriel, une **gouvernance
du modèle explicite** et un **package de preuves publiques** (repo, démo, article) attestant la compétence
data & IA de son auteur.

## 2. Objectifs mesurables (la boussole)

| # | Objectif | Indicateur | Cible |
|---|---|---|---|
| O1 | Démontrer un pipeline MLOps de bout en bout | Pipeline reproductible `make pipeline` sur machine vierge | 100 % succès, zéro étape manuelle |
| O2 | Prouver la rigueur de la méthode | Gates qualité branchées en CI (ruff, mypy --strict, tests, coverage) | 100 % vert sur `main` |
| O3 | Garantir la traçabilité du modèle | Registre MLflow + model card + historique de décisions | ≥ 1 modèle enregistré, 4 tables d'audit peuplées |
| O4 | Surveiller le modèle en production | Drift (Evidently), alertes seuils (ROC-AUC, ECE, PSI), métriques Prometheus/Grafana | Rapport automatisé + alertes objectivables |
| O5 | Fournir une preuve cliquable | Démo en ligne (Hugging Face Spaces) | Lien public fonctionnel, < 5 s de chargement |
| O6 | Valoriser le travail publiquement | Repo vitrine + article LinkedIn + captures | Publication live, 1 lien repo, 1 lien démo |
| O7 | Tenir la honnêteté scientifique | Transparence sur données synthétiques (jamais présentées comme réelles) | Mention explicite dans README, model card, article |

## 3. Périmètre

**Inclus**
- Pipeline de données versions (DVC) : synthèse → features officielles (25) → entraînement → évaluation → décision.
- Modèle XGBoost calibré (isotonic/Platt), registre MLflow, stages Staging/Production/Archived.
- API FastAPI (JWT, rate limiting, schémas stricts), base PostgreSQL/audit (Alembic).
- Monitoring : Evidently (drift), PSI, alertes ; Prometheus + Grafana.
- Déploiement : Docker, GitHub Actions (CI/CD + GHCR), Terraform + K8s (cible prod Oracle Always-Free), démo HF Spaces.
- Preuves : repo public soigné, démo cliquable, article LinkedIn.

**Hors périmètre (explicite)**
- Données réelles CIF (jamais eu accès → synthétique, assumé).
- Décisions de crédit contractuelles (le système est un outil d'aide, human-in-the-loop obligatoire).
- Multi-tiers / production 24/7 avec SLO payant (cible désignée mais non budgétée).

## 4. Normes de référence (pourquoi, pas par habitude)

| Pratique | Justification |
|---|---|
| TDSP / CRISP-DM | Cycle structuré : compréhension → préparation → modélisation → validation → déploiement → suivi |
| MLOps L1 (Google) | CI/CD du code *et* du modèle + pipeline reproductible + registre |
| DVC | Versionner données + modèles avec git (léger, open source, standard) |
| MLflow registry | Traçabilité des versions, stages, transition documentée |
| Evidently + PSI | Détection de dérive données/modèle (le modèle dérive même sans erreur de code) |
| Gates de qualité | ruff + mypy `--strict` + pytest + coverage ≥ 90 % sur CI : le code sale ne passe pas |
| Scripts make one-command | Reproductibilité totale, zéro dépendance à la mémoire humaine |

## 5. Gouvernance du modèle

- **Transparence** : model card (usage prévu, population cible, métriques, limites, équité) générée et versionnée.
- **Honnêteté** : le ROC-AUC (synthétique) n'est **jamais** présenté comme un benchmark réel CIF.
- **Décision** : model ≠ policy ≠ workflow ≠ décision (§64 du cahier) ; seuils documentés ; revue humaine pour REVUE_HUMAINE.
- **Auditabilité** : chaque prédiction, jeton émis et changement enregistré en base (customers, predictions, audit_log, model_versions).
- **Monitoring** : alertes à seuils fixes (ROC-AUC < 0.65, ECE > 0.10, PSI > 0.25, drift ratio ≥ 0.20) + dashboards.

## 6. Architecture cible

```
Clients (Agents CIF) → Cloudflare (WAF/HTTPS) → Ingress Nginx → API FastAPI (2 pods)
                                                          ├─ PostgreSQL 16 (audit + registry)
                                                          ├─ MLflow (tracking/registry)
                                                          └─ Prometheus + Grafana (métriques)
                                                             + Evidently (drift) + Sentry (erreurs)
```

## 7. Roadmap — phases, livrables, critères d'acceptation

| Phase | Livrable | Critère d'acceptation |
|---|---|---|
| P0 — Fondations & gouvernance | Charter, structure repo unique, README vitrine | Repo public propre, historique exploitable |
| P1 — Méthode reproductible | DVC pipeline (prepare→train→evaluate), garde anti-leakage, features officielles | `make pipeline` reproductible + tests verts |
| P2 — Modèle & registre | XGBoost calibré, MLflow registry, model card | 1 modèle enregistré (stage défini), artefacts visibles |
| P3 — API & données | FastAPI JWT, schémas stricts, PostgreSQL/Alembic, audit trail | /docs OK, 100 % prédictions journalisées |
| P4 — Monitoring | Evidently drift + PSI + alertes, Prometheus/Grafana | Rapport automatisé + seuils testés |
| P5 — Déploiement | Docker, CI/CD GHCR, démo HF Spaces, Terraform/K8s (cible) | Image buildée, démo en ligne, deploy scripté |
| P6 — Preuves & visibilité | Repo vitrine, captures, article LinkedIn publié | 1 lien repo + 1 lien démo + article live |

## 8. KPIs fortement liés aux objectifs

- Couverture de tests ≥ 90 % sur les modules critiques (API, monitoring, features).
- mypy `--strict` : 0 erreurs ; ruff : 0 warnings.
- P95 de latence API < 100 ms (charge locust 100 req/s).
- Démo utilisable public : score + décision + explication en < 5 s.

## 9. Risques & atténuations

| Risque | Impact | Atténuation |
|---|---|---|
| Données réelles indisponibles | Métriques non représentatives | Assumer clairement ; proof = pipeline, pas le chiffre |
| Drift en prod | Décisions dégradées silencieusement | Evidently + alertes + dashboards directs |
| Copie du code sans contexte | Incompréhension | README vitrine + Makefile one-command + charter |
| Fausse impression de « modèle réel » | Crédibilité détruite | Mention synthétique partout (README, model card, article) |

## 10. Preuve LinkedIn (objectifs & mesure)

- **Objectif** : susciter reconnaissance + opportunités (pas du vanity).
- **Contenu** : histoire (problème → méthode → système → résultats honnêtes) + liens repo/démo/captures.
- **Mesures d'impact** : impressions, réactions, commentaires, connexions/issues reçues, vues du repo.
- **Format** : post technique structuré (cf. docs/LINKEDIN_POST.md) publié après validation de la démo.