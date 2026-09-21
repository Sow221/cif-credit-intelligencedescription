"""Recherche d'hyperparamètres XGBoost (Optuna) avec validation croisée *temporelle*.

Les plis sont des fenêtres expansives dans l'ordre des dates : on apprend toujours sur le passé
pour prédire le futur. Le jeu de validation et le jeu de test ne sont jamais utilisés ici.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from data.lending_club import DATE_COL, TARGET
from features.lending_club import FEATURES, MONOTONE_CONSTRAINTS
from models.scoring import make_preprocessor


def tune_xgboost(
    train: pd.DataFrame,
    n_trials: int,
    max_rows: int,
    seed: int = 42,
    n_splits: int = 3,
) -> tuple[dict[str, Any], float]:
    """Retourne les meilleurs hyperparamètres et l'AUC moyen en validation croisée temporelle."""
    sample = train
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=seed)
    sample = sample.sort_values(DATE_COL).reset_index(drop=True)

    pre = make_preprocessor("tree")
    x = pre.fit_transform(sample[list(FEATURES)])  # encodage non supervisé : aucune fuite de cible
    names = list(pre.get_feature_names_out())
    constraints = tuple(MONOTONE_CONSTRAINTS.get(n, 0) for n in names)
    y = sample[TARGET].to_numpy()
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(x))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        scores: list[float] = []
        for tr_idx, va_idx in folds:
            clf = XGBClassifier(
                **params,
                monotone_constraints=constraints,
                tree_method="hist",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            )
            clf.fit(x[tr_idx], y[tr_idx])
            scores.append(float(roc_auc_score(y[va_idx], clf.predict_proba(x[va_idx])[:, 1])))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params), float(study.best_value)
