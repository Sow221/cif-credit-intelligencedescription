"""CLI : cif-replay-monitoring — rejoue la surveillance sur l'historique Lending Club, tracé MLflow."""

from __future__ import annotations

import click
import mlflow
import pandas as pd

from config import load_config
from data.lending_club import temporal_partition
from models.lending_club_registry import DEFAULT_MODEL_NAME
from monitoring.replay import run_replay, write_replay_artifacts
from utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--out", "out_dir", type=str, default="reports/lending_club_monitoring", help="Dossier de sortie.")
@click.option("--tracking-uri", type=str, default=None)
@click.option("--model-uri", type=str, default=f"models:/{DEFAULT_MODEL_NAME}@champion")
def main(out_dir: str, tracking_uri: str | None, model_uri: str) -> None:
    """Rejoue le monitoring (dérive + performance retardée) sur validation+test, champion enregistré."""
    cfg = load_config()
    lc = cfg.lending_club
    features = pd.read_parquet(lc.processed_path)
    train, _, _ = temporal_partition(features, lc)

    mlflow.set_tracking_uri(tracking_uri or cfg.model.mlflow_tracking_uri)
    loaded = mlflow.pyfunc.load_model(model_uri)
    champion = loaded.unwrap_python_model().model

    mlflow.set_experiment("lending_club_monitoring_replay")
    with mlflow.start_run(run_name="monitoring_replay"):
        result = run_replay(features, train, champion)
        out = write_replay_artifacts(result, features, train, out_dir)
        for _, row in result.drift_series.iterrows():
            mlflow.log_metric("psi", row["psi"], step=int(row["month"].replace("-", "")))
        for _, row in result.performance_series.iterrows():
            step = int(row["month"].replace("-", ""))
            mlflow.log_metric("realized_roc_auc", row["roc_auc"], step=step)
            mlflow.log_metric("realized_ece", row["ece"], step=step)
        mlflow.log_metric("any_alert", int(result.any_alert))
        mlflow.log_artifacts(str(out))

    logger.info("cli.replay_monitoring.done", any_alert=result.any_alert, out=str(out))
    print(f"Rejeu terminé — alerte déclenchée : {result.any_alert}. Rapports dans {out}")
