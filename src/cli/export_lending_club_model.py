"""CLI : cif-export-lc-model — exporte le champion Lending Club pour Docker/HF Spaces.

Copie l'artefact pyfunc déjà enregistré (``models:/lending_club_champion@champion``) en dossier
autonome, chargeable sans serveur MLflow (``mlflow.pyfunc.load_model(chemin_local)``). Le seuil
de décision (posé en tag sur la version dans le registre) est copié à côté en fichier texte,
puisque le tag lui-même n'est lisible qu'en interrogeant un registre — absent en production
HF Spaces (voir ``services.lending_club_predictor.LendingClubPredictor.from_registry``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import click
import mlflow

from models.lending_club_registry import DEFAULT_MODEL_NAME
from utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--output", type=str, default="deploy/model_lending_club", help="Dossier de sortie.")
@click.option("--model-uri", type=str, default=f"models:/{DEFAULT_MODEL_NAME}@champion")
@click.option("--tracking-uri", type=str, default=None)
def main(output: str, model_uri: str, tracking_uri: str | None) -> None:
    """Exporte le champion Lending Club enregistré en artefact autonome + seuil de décision."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    out_dir = Path(output)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    local_path = mlflow.artifacts.download_artifacts(model_uri)
    shutil.copytree(local_path, out_dir)

    name, alias = model_uri.removeprefix("models:/").split("@")
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    threshold = mv.tags.get("cost_threshold", "0.15")
    (out_dir / "cost_threshold.txt").write_text(threshold, encoding="utf-8")

    logger.info("cli.export_lc_model.done", output=str(out_dir.resolve()), version=mv.version, threshold=threshold)
    print(f"Champion Lending Club v{mv.version} exporté dans {out_dir.resolve()} (seuil={threshold})")


if __name__ == "__main__":
    main()
