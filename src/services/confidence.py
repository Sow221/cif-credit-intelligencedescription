"""Calcul de la confiance d'une prédiction — exigence cabinet (Semaine 2, src/services/confidence.py).

La confiance exprime le degré de certitude de la décision : une probabilité *éloignée* de toutes
les frontières de décision (seuils de la policy) est décidée avec une confiance élevée ; proche
d'un seuil, la confiance chute (le cas est « ambiguous » et passera en revue humaine).

Cette mesure est déterministe et entièrement traçable — elle alimente le champ ``confidence``
de la réponse API et de l'audit trail (§M07).
"""

from __future__ import annotations

import numpy as np

DecisionThresholds = tuple[float, float, float]


def prediction_confidence(
    probability: float,
    thresholds: DecisionThresholds = (0.10, 0.25, 0.50),
    margin: float = 0.10,
) -> float:
    """Confiance dans [0, 1] basée sur la distance au seuil de décision le plus proche.

    Args:
        probability: probabilité estimée de défaut.
        thresholds: seuils (approbation, revue, rejet) de la policy.
        margin: largeur de référence pour normaliser la distance (par défaut 0.10).

    Returns:
        Confiance entre 0.0 (frontière exacte) et 1.0 (maximale).
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability hors bornes : {probability}")
    distances = [abs(probability - threshold) for threshold in thresholds]
    min_distance = min(distances or [1.0])
    return float(np.clip(min_distance / max(margin, 1e-6), 0.0, 1.0).round(6))


def confidence_rating(confidence: float) -> str:
    """Qualifie la confiance pour le reporting (élevée / moyenne / faible)."""
    if confidence >= 0.66:
        return "elevee"
    if confidence >= 0.33:
        return "moyenne"
    return "faible"
