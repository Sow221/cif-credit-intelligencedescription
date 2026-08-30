"""Pipeline d'entraînement continu (Continuous Training) — Dagster.

Orchestration standard de la boucle MLOps : génération des données -> features ->
entraînement (MLflow) -> validation -> promotion de la meilleure version dans le
registry (alias ``production``). Réutilise les fonctions existantes (aucun duplicata) :
``data.synthetic.generate_datasets``, ``features.builder.build_features``,
``models.train.train_and_log``. Un schedule quotidien relance la boucle.

Usage :
    dagster dev -m pipelines.continuous_training
    # ou en prod : dagster job execute continuous_training (via dagster-cloud / cron)
"""

from __future__ import annotations

import os

import mlflow
from dagster import Definitions, asset, define_asset_job, schedule

from config.schema import DataConfig, FeatureConfig, ModelConfig
from data.synthetic import GeneratedDatasets, generate_datasets
from features.builder import build_features
from models.train import train_and_log

MODEL_NAME = "cif_credit_official"
PROMOTION_ROC_AUC_MIN = 0.65


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")


@asset(description="Génération du jeu de données (synthétique, honnête labellisage)")
def raw_datasets() -> GeneratedDatasets:
    return generate_datasets(DataConfig())


@asset(description="Construction de la matrice des 25 features officielles")
def feature_table(raw_datasets: GeneratedDatasets) -> object:
    return build_features(raw_datasets.customers, raw_datasets.loans, raw_datasets.savings, FeatureConfig())


@asset(description="Entraînement + logging MLflow + enregistrement au registry")
def trained_model(feature_table: object) -> dict:
    mlflow.set_tracking_uri(_tracking_uri())
    result = train_and_log(
        feature_table,
        ModelConfig(),
        FeatureConfig(),
        run_name="continuous_training",
        register=True,
    )
    return {"run_id": result.run_id, "metrics": result.metrics}


@asset(description="Validation selon le seuil ROC-AUC cabinet (>= 0.65)")
def validated_model(trained_model: dict) -> dict:
    auc = float(trained_model["metrics"].get("roc_auc", 0.0))
    return {"passed": auc >= PROMOTION_ROC_AUC_MIN, "roc_auc": auc}


@asset(description="Promotion vers l'alias 'production' si validation OK")
def promoted_model(validated_model: dict, trained_model: dict) -> dict:
    if not validated_model["passed"]:
        return {"promoted": False, "reason": "roc_auc < seuil"}
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(_tracking_uri())
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, "production", latest.version)
    return {"promoted": True, "version": latest.version}


continuous_training_job = define_asset_job(
    "continuous_training",
    [raw_datasets, feature_table, trained_model, validated_model, promoted_model],
)


@schedule(job=continuous_training_job, cron_schedule="0 3 * * *", description="Réentraînement quotidien à 03:00")
def continuous_training_schedule() -> dict:
    return {}


definitions = Definitions(
    assets=[raw_datasets, feature_table, trained_model, validated_model, promoted_model],
    schedules=[continuous_training_schedule],
)
