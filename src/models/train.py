"""Entraînement des modèles avec tracking MLflow et registry versionné."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from config.schema import FeatureConfig, ModelConfig
from evaluation.metrics import compute_all_metrics
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Résultat d'un entraînement."""

    model: Any
    metrics: dict[str, float]
    run_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


def feature_columns(cfg: FeatureConfig) -> list[str]:
    """Retourne la liste ordonnée des 25 features (source de vérité partagée)."""
    cols: list[str] = []
    for family in cfg.families.values():
        for col in family:
            if col not in cols:
                cols.append(col)
    return cols


def temporal_split(
    df: pd.DataFrame,
    cfg: ModelConfig,
    target: str,
    *,
    order_col: str = "customer_id",
    split_date: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split TEMPOREL (protocole CIF) : l'ordre des lignes est l'ordre du temps.

    Remplace le split aléatoire (interdit par la discipline CIF). NB : le jeu
    synthétique ne porte pas de vraie date ; le proxy temporel est l'ordre des
    ``customer_id`` (les premiers contrats sont les plus anciens). À substituer
    par une vraie colonne ``application_date`` sur les données CIF réelles.
    """
    cols = feature_columns(FeatureConfig())
    df = df.sort_values(order_col).reset_index(drop=True)
    cutoff = int(len(df) * split_date)
    train_df = df.iloc[:cutoff].copy()
    val_df = df.iloc[cutoff:].copy()

    X_tr = train_df[cols].astype(float)
    X_te = val_df[cols].astype(float)
    y_tr = train_df[target].astype(int).to_numpy()
    y_te = val_df[target].astype(int).to_numpy()
    return X_tr, X_te, y_tr, y_te


def train_split(
    df: pd.DataFrame, cfg: ModelConfig, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split train/test TEMPOREL reproductible (protocole CIF, jamais aléatoire)."""
    return temporal_split(df, cfg, target)


def make_model(cfg: ModelConfig, scale_pos_weight: float = 1.0) -> Any:
    """Instancie le modèle XGBoost (calibré si activé) avec les hyperparamètres de la config."""
    hp = dict(cfg.xgboost)
    hp["scale_pos_weight"] = scale_pos_weight
    base = XGBClassifier(
        **hp,
        random_state=cfg.random_state,
        eval_metric="aucpr",
    )
    if cfg.calibration.enabled:
        # Calibration isotonique (exigence §93) : prédit des probabilités *bien calibrées*,
        # comparables entre clients et exploitables par le decision engine.
        return CalibratedClassifierCV(
            estimator=base,
            method=cfg.calibration.method,
            cv=cfg.calibration.cv,
            n_jobs=1,
        )
    return base


def train_and_log(
    df: pd.DataFrame,
    model_cfg: ModelConfig,
    feature_cfg: FeatureConfig,
    *,
    experiment_id: str | None = None,
    run_name: str = "xgboost",
    register: bool = False,
    log_artifacts: bool = True,
) -> TrainingResult:
    """Entraîne, évalue et journalise dans MLflow (métriques + signature + artifacts)."""
    cols = feature_columns(feature_cfg)
    X_tr, X_te, y_tr, y_te = temporal_split(df, model_cfg, feature_cfg.target)

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    model = make_model(model_cfg, scale_pos_weight=pos_weight)
    if model_cfg.calibration.enabled:
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    probs = model.predict_proba(X_te)[:, 1]
    metrics = compute_all_metrics(y_te, probs)
    params: dict[str, Any] = {**model_cfg.xgboost, "scale_pos_weight": pos_weight, "test_size": model_cfg.test_size}
    if model_cfg.calibration.enabled:
        params["calibration"] = f"{model_cfg.calibration.method}-cv{model_cfg.calibration.cv}"

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("n_features", len(cols))
        mlflow.log_param("algorithm", "XGBClassifier")
        mlflow.set_tags({"domain": "credit_scoring", "stage": "prototype"})

        if log_artifacts:
            signature = infer_signature(X_te, probs)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_te.iloc[:5],
                registered_model_name="cif_credit_official" if register else None,
                serialization_format="cloudpickle",
                pip_requirements=["xgboost>=2.0", "scikit-learn>=1.4", "numpy>=1.26", "pandas>=2.1"],
            )
            feat_path = Path(model_cfg.artifacts_dir) / "feature_columns.json"
            feat_path.parent.mkdir(parents=True, exist_ok=True)
            feat_path.write_text(json.dumps(cols), encoding="utf-8")
            mlflow.log_artifact(str(feat_path), artifact_path="model_metadata")

        logger.info("models.train.done", run_id=run.info.run_id, metrics=metrics)
        return TrainingResult(model=model, metrics=metrics, run_id=run.info.run_id, params=params)


def train_plain(
    df: pd.DataFrame, model_cfg: ModelConfig, feature_cfg: FeatureConfig
) -> tuple[XGBClassifier, dict[str, float]]:
    """Entraîne sans MLflow (pour tests unitaires et ablation study)."""
    X_tr, X_te, y_tr, y_te = temporal_split(df, model_cfg, feature_cfg.target)
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    model = make_model(model_cfg, scale_pos_weight=pos_weight)
    if model_cfg.calibration.enabled:
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    probs = model.predict_proba(X_te)[:, 1]
    return model, compute_all_metrics(y_te, probs)
