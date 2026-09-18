"""Ablation study par famille de features (audit Phase A/B) — M0..M3."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from config.schema import FeatureConfig, ModelConfig
from utils.logging import get_logger
from utils.seed import seed_everything

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
    from models.train import make_model

    seed_everything(seed)
    families = feature_cfg.families

    target = feature_cfg.target
    ordered = df.sort_values("customer_id").reset_index(drop=True)
    cutoff = int(len(ordered) * 0.8)
    train_df, val_df = ordered.iloc[:cutoff], ordered.iloc[cutoff:]
    y_tr = train_df[target].astype(int).to_numpy()
    y_te = val_df[target].astype(int).to_numpy()

    def _eval(cols: list[str]) -> float:
        # Holdout TEMPOREL (jamais d'évaluation en interne — un modèle évalué sur ses
        # propres données d'entraînement surestime systématiquement sa performance).
        X_tr = train_df[cols].astype(float)
        X_te = val_df[cols].astype(float)
        model = make_model(model_cfg)
        if model_cfg.calibration.enabled:
            model.fit(X_tr, y_tr)
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        probs = model.predict_proba(X_te)[:, 1]
        return float(roc_auc_score(y_te, probs))

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
