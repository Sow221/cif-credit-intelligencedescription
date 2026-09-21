# ADR 0003 — Règle de sélection du champion

**Statut** : accepté

Le modèle complexe (XGBoost contraint) n'est déclaré champion que si l'IC à 95 % **apparié** de
l'écart d'AUC (XGBoost − baseline logistique) sur le test out-of-time exclut 0. Sinon la baseline est
retenue : plus simple, plus lisible, plus facile à justifier à un client refusé.

Autres choix du protocole : hyperparamètres par validation croisée temporelle dans l'entraînement
seul (Optuna) ; calibration isotonique sur la validation ; contraintes de monotonicité sur les
variables au sens économique connu ; pas de `scale_pos_weight` (il fausse les probabilités
alors qu'elles servent à fixer des seuils) ; test évalué une seule fois.
