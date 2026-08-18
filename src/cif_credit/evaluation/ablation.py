"""Ablation study par famille de features (audit Phase A/B) — M0..M3."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from cif_credit.config.schema import FeatureConfig, ModelConfig
from cif_credit.utils.logging import get_logger
from cif_credit.utils.seed import seed_everything

logger = get_logger(__name__)


def run_ablation(
    df: pd.DataFrame,
    model_cfg: ModelConfig,
    feature_cfg: FeatureConfig,
    seed: int = 42,
) -> dict[str, float]:
    """Évalue la contribution de chaque famille de features.

    Modèles :
    - M0_BASELINE      : aucune feature (performance aléatoire attendue ~0.5)
    - M1_PROFIL_INCOME : features profil + revenu
    - M2_SAVINGS       : M1 + famille épargne
    - M3_HISTORY       : M2 + famille historique (jeu complet)
    """
    from cif_credit.models.train import make_model

    seed_everything(seed)
    families = feature_cfg.families

    target = feature_cfg.target
    y = df[target].astype(int).to_numpy()

    def _eval(cols: list[str]) -> float:
        X = df[cols].astype(float)
        model = make_model(model_cfg)
        model.fit(X, y, eval_set=[(X, y)], verbose=False)
        probs = model.predict_proba(X)[:, 1]
        return float(roc_auc_score(y, probs))

    profile = families["profile_income"]
    savings = profile + families["savings"]
    history = savings + families["history"]

    results = {
        "M0_BASELINE": 0.5,
        "M1_PROFIL_INCOME": _eval(profile),
        "M2_SAVINGS": _eval(savings),
        "M3_HISTORY": _eval(history),
    }
    logger.info("evaluation.ablation.done", results=results)
    return results
