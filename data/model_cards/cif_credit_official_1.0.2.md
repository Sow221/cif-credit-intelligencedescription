# Model Card — cif_credit_official
**Version** : 1.0.2 | **Type** : classification_binaire | **Algorithme** : XGBClassifier + CalibratedClassifierCV(isotonic)
**Responsable** : TELQAN / CIF Credit Intelligence (contact@cif-digital-platform.test)

## Usage prévu
Scoring de crédit (estimation de la probabilité de défaut) des clients CIF/DigiCoop-WA+ ; décision finale sous revue humaine (human-in-the-loop) obligatoire.

## Population cible
Détenteurs de produits CIF/DigiCoop-WA+ (pilot synthétique et réel).

## Métriques d'entraînement
| Métrique | Valeur |
|---|---|
| brier | 0.0626 |
| calibration_intercept | -0.0056 |
| calibration_slope | 0.9951 |
| ece | 0.0076 |
| log_loss | 0.2021 |
| pr_auc | 0.6334 |
| roc_auc | 0.9246 |

## Seuils de décision (Policy)
| Seuil | Valeur |
|---|---|
| approve | 0.1000 |
| hard_reject | 0.5000 |
| review | 0.2500 |

## Surveillance (Monitoring)
| Alerte | Valeur |
|---|---|
| ece_max | 0.1 |
| psi_max | 0.25 |
| roc_auc_min | 0.65 |

## Limites
- Modèle entraîné sur données synthétiques pour le pilot.
- Pas de décision contractuelle sans revue humaine (REVUE_HUMAINE).
- À ré-entraîner sur les données réelles CIF (protocole shadow mode).
- Le générateur synthétique dérive toutes les features d'un facteur latent unique : le ROC-AUC observé surestime structurellement la séparabilité attendue sur données réelles, multi-causales et non stationnaires. À réévaluer sur données CIF réelles.
- Split temporel désormais fondé sur la vraie date de demande (application_date) avec jointure d'historique point-in-time stricte (aucun prêt daté à/après la demande n'entre dans l'agrégat) — corrige un split précédent ordonné par customer_id (proxy non temporel). Le ROC-AUC baisse légèrement (0.940 -> 0.925) : c'est la mesure honnête, pas une régression du modèle.

## Considérations d'équité
Audit d'équité à reproduire sur données réelles (disparités par profil non fiabilisées sur données synthétiques).

## Sources de données
- Prototype synthétique TELQAN (audit phases A-F)

*Générée le 2026-09-21T09:26:07.969338+00:00.*
