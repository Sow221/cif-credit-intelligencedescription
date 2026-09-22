"""Modèles de scoring sur features tabulaires : baseline logistique et XGBoost contraint.

Chaque modèle est un objet autosuffisant (prétraitement + estimateur + calibration isotonique)
qui expose ``predict_proba`` et, pour XGBoost, ``explain`` (contributions par feature d'origine,
calcul natif exact via ``pred_contribs`` — sans dépendance externe).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import DMatrix, XGBClassifier

from features.lending_club import CATEGORICAL_FEATURES, FEATURES, MONOTONE_CONSTRAINTS, NUMERIC_FEATURES

PROB_EPS = 1e-4


def make_preprocessor(kind: str) -> ColumnTransformer:
    """Prétraitement adapté au modèle : imputation + standardisation (linéaire) ou NaN natifs (arbres)."""
    if kind == "linear":
        numeric: Any = Pipeline(
            [("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]
        )
    else:
        numeric = "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        verbose_feature_names_out=False,
    )


@dataclass
class ScoringModel:
    """Prétraitement + estimateur + calibrateur, sérialisable d'un bloc."""

    name: str
    preprocessor: ColumnTransformer
    estimator: Any
    calibrator: IsotonicRegression | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(df[list(FEATURES)])
        return np.asarray(self.estimator.predict_proba(matrix)[:, 1], dtype=float)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilité de défaut (calibrée si un calibrateur est présent)."""
        p = self.raw_proba(df)
        if self.calibrator is not None:
            p = np.asarray(self.calibrator.predict(p), dtype=float)
        return np.clip(p, PROB_EPS, 1.0 - PROB_EPS)

    def calibrate(self, df: pd.DataFrame, y: np.ndarray) -> ScoringModel:
        """Ajuste la calibration isotonique sur un jeu *distinct* de l'entraînement."""
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(self.raw_proba(df), np.asarray(y, dtype=int))
        self.calibrator = iso
        return self

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Contributions (log-odds, avant calibration) par feature d'origine ; colonne ``bias`` en plus.

        Positif = augmente le risque de défaut. Les colonnes dérivées (indicatrices de valeur
        manquante, one-hot des catégorielles) sont réagrégées sur leur variable source
        (ex. ``purpose_*`` → ``purpose``, ``missingindicator_mort_acc`` → ``mort_acc``).
        Exact pour les deux modèles supportés : somme des contributions + biais = la sortie
        brute du modèle (avant calibration), vérifié par ``test_explanations_sum_to_model_margin``.
        """
        names = list(self.preprocessor.get_feature_names_out())
        matrix = self.preprocessor.transform(df[list(FEATURES)])

        if isinstance(self.estimator, XGBClassifier):
            contribs = self.estimator.get_booster().predict(DMatrix(matrix, feature_names=names), pred_contribs=True)
            frame = pd.DataFrame(contribs, columns=[*names, "bias"], index=df.index)
        elif isinstance(self.estimator, LogisticRegression):
            # Linéaire et exact : contribution_i = coef_i * valeur_i (espace prétraité), le
            # score brut est leur somme + intercept — pas d'approximation, contrairement à un
            # explicateur générique (SHAP/LIME).
            coef = self.estimator.coef_[0]
            contrib_matrix = np.asarray(matrix, dtype=float) * coef
            frame = pd.DataFrame(contrib_matrix, columns=names, index=df.index)
            frame["bias"] = float(self.estimator.intercept_[0])
        else:
            raise NotImplementedError(f"explain() non défini pour {type(self.estimator).__name__}")

        return _group_by_source_feature(frame)


def _group_by_source_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Réagrège les colonnes dérivées (one-hot, indicatrices manquantes) sur leur feature d'origine."""
    grouped: dict[str, pd.Series | float] = {}
    for col in frame.columns:
        if col == "bias":
            source = "bias"
        elif col.startswith("missingindicator_"):
            source = col.removeprefix("missingindicator_")
        else:
            source = next((c for c in CATEGORICAL_FEATURES if col.startswith(f"{c}_")), col)
        grouped[source] = grouped.get(source, 0.0) + frame[col]
    return pd.DataFrame(grouped)


def fit_logistic(train: pd.DataFrame, y: np.ndarray, c: float = 1.0) -> ScoringModel:
    """Baseline : régression logistique régularisée (référence historique du scoring)."""
    pre = make_preprocessor("linear")
    x = pre.fit_transform(train[list(FEATURES)])
    clf = LogisticRegression(C=c, max_iter=2000)
    clf.fit(x, y)
    return ScoringModel("logistic", pre, clf, params={"C": c})


def fit_xgboost(train: pd.DataFrame, y: np.ndarray, params: dict[str, Any], seed: int = 42) -> ScoringModel:
    """XGBoost avec contraintes de monotonicité sur les variables à sens économique connu.

    Pas de ``scale_pos_weight`` : il fausserait les probabilités, or elles servent à fixer des seuils.
    """
    pre = make_preprocessor("tree")
    x = pre.fit_transform(train[list(FEATURES)])
    names = list(pre.get_feature_names_out())
    constraints = tuple(MONOTONE_CONSTRAINTS.get(n, 0) for n in names)
    clf = XGBClassifier(
        **params,
        monotone_constraints=constraints,
        tree_method="hist",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x, y)
    return ScoringModel("xgboost", pre, clf, params=dict(params))
