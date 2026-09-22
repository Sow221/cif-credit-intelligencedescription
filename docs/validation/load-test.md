# Test de charge — `/v1/lending-club/score`

**Date** : 2026-09-22 · **Environnement** : codespace de développement partagé (CPU mutualisé
avec Dagster, MLflow et Docker tournant en parallèle) — **pas** un nœud de production dédié.
Les chiffres ci-dessous donnent un ordre de grandeur, pas un SLO garanti.

## Méthode

Script `asyncio` + `httpx`, 2 workers uvicorn, modèle `lending_club_champion` chargé depuis un
artefact local (même configuration que l'image Docker de démo). Chaque scénario envoie N
requêtes concurrentes identiques et mesure la latence de bout en bout (JWT déjà obtenu, hors
mesure).

## Résultat 1 — le rate limiting fonctionne (comportement voulu)

Avec la limite par défaut (120 req/min/client), 1000 requêtes envoyées en rafale par un seul
client : **235 acceptées, 765 rejetées en 429**, cohérent avec la fenêtre glissante de 60 s
configurée dans `api/middleware.py`. C'est la protection anti-abus qui marche, pas un défaut.

## Résultat 2 — capacité brute (limite temporairement levée)

| Scénario | Requêtes | Succès | Débit | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| Charge légère (5 concurrentes) | 100 | 100/100 | 90,5 req/s | 35,9 ms | 91,3 ms | 95,0 ms | 95,1 ms |
| Charge soutenue (50 concurrentes) | 1000 | 1000/1000 | 93,6 req/s | 517,5 ms | 652,3 ms | 811,9 ms | 868,5 ms |

Aucune erreur ni crash sous charge soutenue. Le débit plafonne autour de **~90 req/s** dans cet
environnement (2 workers, CPU partagé) : au-delà de quelques requêtes concurrentes, la latence
absorbe la charge plutôt que le débit — comportement sain (pas de défaillance en cascade), mais
la capacité réelle en production dépendra du nombre de cœurs alloués aux pods.

## Bug découvert et corrigé pendant ce test

`LendingClubScoreRequest.features` n'acceptait que `float | str`, jamais `null`. Un client HTTP
strict (`httpx`) refuse même d'émettre une requête contenant un `NaN` littéral (JSON non
standard) — or c'est ce que produisait naïvement toute tentative d'envoyer une valeur manquante.
Le modèle sait gérer l'absence d'une variable (imputation), mais l'API n'offrait aucun moyen
valide de le lui dire. Corrigé : le schéma accepte désormais `null` explicitement
(`float | str | None`), verrouillé par `test_lending_club_score_accepts_null_for_missing_numeric_feature`.

## Limites de cette mesure

- Un seul type de requête (Lending Club), une seule forme de payload.
- Pas de montée en charge progressive (ramp-up), pas de test de durée (soak test).
- Environnement de développement, pas les ressources K8s ciblées (`k8s/deployment.yaml` :
  250m-500m CPU par pod, 2 réplicas) — à revalider sur la cible réelle avant tout engagement de SLO.
