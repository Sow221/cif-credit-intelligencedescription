"""CLI : cif-train — entraînement XGBoost, tracking MLflow et registration."""

from __future__ import annotations

import click
import mlflow
import pandas as pd

from cif_credit.config import load_config
from cif_credit.models.train import train_and_log
from cif_credit.utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--register/--no-register", default=True, help="Enregistre le modèle dans le registry.")
@click.option("--tracking-uri", type=str, default=None, help="URI MLflow (ex: sqlite:///mlruns.db).")
@click.option("--experiment", type=str, default=None, help="Nom de l'expérience MLflow.")
@click.option("--data", type=str, default=None, help="Chemin du fichier de features parquet (défaut: config).")
def main(
    register: bool,
    tracking_uri: str | None,
    experiment: str | None,
    data: str | None,
) -> None:
    """Entraîne le modèle officiel et journalise dans MLflow."""
    cfg = load_config()

    path = data or f"{cfg.data.processed_dir}/{cfg.features.output_file}"
    df = pd.read_parquet(path)

    mlflow.set_tracking_uri(tracking_uri or cfg.model.mlflow_tracking_uri)
    mlflow.set_experiment(experiment or cfg.model.mlflow_experiment)

    result = train_and_log(
        df,
        cfg.model,
        cfg.features,
        register=register,
        run_name=f"xgboost_{cfg.model.algorithm}",
    )

    logger.info(
        "cli.train.done",
        run_id=result.run_id,
        roc_auc=result.metrics["roc_auc"],
        pr_auc=result.metrics["pr_auc"],
        brier=result.metrics["brier"],
        registered=register,
    )
