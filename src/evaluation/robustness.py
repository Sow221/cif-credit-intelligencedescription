"""Analyse de robustesse : dégradation du ROC-AUC sous bruit de features (audit Phase B)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from utils.logging import get_logger

logger = get_logger(__name__)


def robustness_curve(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    noise_levels: list[float] | None = None,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Ajoute un bruit gaussien croissant aux features et mesure le ROC-AUC.

    Permet de détecter un seuil critique où la performance s'effondre.
    """
    noise_levels = noise_levels or [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    rng = np.random.default_rng(seed)
    levels: list[float] = []
    aucs: list[float] = []

    X_float = X_test.astype(float)
    std = X_float.std().replace(0, 1.0)

    for level in noise_levels:
        noise = rng.normal(0, level, size=X_float.shape) * std.to_numpy()
        X_noisy = X_float + noise
        probs = model.predict_proba(X_noisy)[:, 1]
        aucs.append(float(roc_auc_score(y_test, probs)))
        levels.append(level)

    logger.info("evaluation.robustness.done", levels=levels, roc_aucs=aucs)
    return {"noise_levels": levels, "roc_aucs": aucs}
