# ADR 0002 — Exclusion de grade, sub_grade et int_rate

**Statut** : accepté

`grade`, `sub_grade` et `int_rate` sont la sortie du scoring interne de Lending Club, connue à
l'octroi. Les utiliser reviendrait à empiler un modèle sur le modèle du prêteur : l'AUC monterait,
mais sans rien démontrer sur la méthode et sans équivalent chez un prêteur qui construit son
premier score. Ils sont exclus par défaut (`lending_club.exclude_lender_score`). Le passer à
`false` permet de mesurer l'apport, à documenter comme tel.
