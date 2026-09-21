"""Choix des seuils de décision par coût économique (et non « à la main »)."""

from __future__ import annotations

import numpy as np


def optimal_cost_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_false_negative: float,
    cost_false_positive: float,
    n_candidates: int = 200,
) -> tuple[float, float]:
    """Seuil de PD minimisant ``c_fn * FN + c_fp * FP`` (refus si PD >= seuil).

    FN = mauvais payeur accepté (perte), FP = bon payeur refusé (manque à gagner).
    Retourne ``(seuil, coût moyen par dossier)``.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, n_candidates)))
    best_t, best_cost = float(candidates[0]), float("inf")
    for t in candidates:
        refuse = p >= t
        fn = int(((~refuse) & (y == 1)).sum())
        fp = int((refuse & (y == 0)).sum())
        cost = (cost_false_negative * fn + cost_false_positive * fp) / len(y)
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t, best_cost
