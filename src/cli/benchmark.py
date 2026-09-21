"""CLI : cif-benchmark — protocole out-of-time complet, tracé dans MLflow."""

from __future__ import annotations

from pathlib import Path

import click
import mlflow
import pandas as pd

from config import load_config
from pipelines.benchmark import run_benchmark, write_artifacts
from utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--trials", type=int, default=None, help="Essais Optuna (défaut : conf lending_club.tuning_trials).")
@click.option("--bootstrap", "n_bootstrap", type=int, default=200, help="Réplications bootstrap des IC95%.")
@click.option("--out", "out_dir", type=str, default="reports/lending_club", help="Dossier de sortie des rapports.")
@click.option("--tracking-uri", type=str, default=None)
def main(trials: int | None, n_bootstrap: int, out_dir: str, tracking_uri: str | None) -> None:
    """Compare baseline et XGBoost sur le test out-of-time et journalise le tout."""
    cfg = load_config()
    lc = cfg.lending_club
    features = pd.read_parquet(lc.processed_path)

    mlflow.set_tracking_uri(tracking_uri or cfg.model.mlflow_tracking_uri)
    mlflow.set_experiment("lending_club_benchmark")
    with mlflow.start_run(run_name="out_of_time_benchmark"):
        result = run_benchmark(features, lc, n_trials=trials, n_bootstrap=n_bootstrap)
        out = write_artifacts(result, out_dir)
        for name, metrics in result.metrics["test_metrics"].items():
            mlflow.log_metrics({f"test_{name}_{k}": v for k, v in metrics.items()})
        mlflow.log_params({"champion": result.metrics["champion"], **{f"lc_{k}": v for k, v in vars(lc).items()}})
        mlflow.log_artifacts(str(out))
    logger.info("cli.benchmark.done", champion=result.metrics["champion"], out=str(Path(out_dir)))
