"""CLI : cif-promote / cif-rollback — gouvernance des versions du modèle servi."""

from __future__ import annotations

import click
import mlflow
from mlflow.tracking import MlflowClient

from config import load_config
from models.promotion import PromotionGates, latest_version, promote_if_better, rollback
from models.registry import DEFAULT_MODEL_NAME


def _client(tracking_uri: str | None) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri or load_config().model.mlflow_tracking_uri)
    return MlflowClient()


@click.command()
@click.option("--model-name", default=DEFAULT_MODEL_NAME)
@click.option("--version", default=None, help="Version candidate (défaut : la plus récente).")
@click.option("--min-gain", type=float, default=0.0, help="Gain minimal d'AUC exigé sur le champion.")
@click.option("--tracking-uri", default=None)
def promote(model_name: str, version: str | None, min_gain: float, tracking_uri: str | None) -> None:
    """Promeut la version candidate au rang de champion si elle franchit les portes de qualité."""
    client = _client(tracking_uri)
    decision = promote_if_better(
        client, model_name, version or latest_version(client, model_name), PromotionGates(min_gain=min_gain)
    )
    click.echo(f"v{decision.version}: {'PROMUE' if decision.promoted else 'REFUSÉE'} — {decision.reason}")
    if not decision.promoted:
        raise SystemExit(1)


@click.command("rollback")
@click.option("--model-name", default=DEFAULT_MODEL_NAME)
@click.option("--tracking-uri", default=None)
def rollback_cmd(model_name: str, tracking_uri: str | None) -> None:
    """Rétablit la version précédente comme champion."""
    click.echo(f"Champion restauré : v{rollback(_client(tracking_uri), model_name)}")
