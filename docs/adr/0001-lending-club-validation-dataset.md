# ADR 0001 — Lending Club comme jeu de validation de la méthode

**Statut** : accepté · **Date** : 2026-09-21

## Contexte
Aucune donnée réelle CIF n'est disponible. Le jeu synthétique historique dérive toutes ses variables
d'un facteur latent unique : son ROC-AUC (≈ 0,94) reflète le générateur, pas une capacité de prédiction.
Il ne permet ni split temporel réel ni comparaison honnête à une baseline.

## Décision
Valider la **méthode** (pas la performance CIF) sur le jeu public Lending Club : vraies dates d'octroi,
issues réelles, volumétrie suffisante. Le générateur synthétique est conservé comme jeu de test.

## Périmètre
Prêts à 36 mois émis entre 2009-01 et 2015-12, statuts terminaux uniquement (remboursé / perte).
Sur un fichier arrêté au T4 2018, tous ces prêts sont arrivés à échéance : le label est mature.
Inclure des prêts à 60 mois ou plus récents introduirait une censure à droite (les pertes se
matérialisent avant les remboursements complets) et biaiserait le taux de défaut.

## Découpage
Entraînement ≤ 2013 ; validation 2014 ; test 2015 (jamais utilisé pour les choix).

## Conséquences
- Résultats publiables comme « méthode validée sur données publiques réelles ».
- Ils **ne** disent rien de la performance sur le portefeuille CIF (crédit américain entre particuliers).
- Réentraînement sur données CIF via le même pipeline dès qu'elles existent.
