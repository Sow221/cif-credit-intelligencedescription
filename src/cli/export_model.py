"""CLI : cif-export-model — exporte un modèle autonome (sans registry MLflow) pour Docker/HF Spaces.

Contrairement à `cif-register-model` (registry MLflow + model card, nécessite un tracking
server), cette commande produit un dossier MLflow "flavor" autosuffisant (``deploy/model/``)
chargeable directement par ``mlflow.sklearn.load_model(path)`` sans aucune dépendance réseau —
c'est l'artefact "cuit" dans l'image Docker de l'API (voir ``deploy/huggingface/Dockerfile``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import click
import mlflow

from config import load_config
from data.synthetic import generate_datasets
from features.builder import build_features
from models.train import train_plain
from utils.logging import get_logger

logger = get_logger(__name__)

# Types non-sklearn requis par la sérialisation skops (calibration isotonique + XGBoost).
_SKOPS_TRUSTED_TYPES = [
    "sklearn.calibration._CalibratedClassifier",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


@click.command()
@click.option("--output", type=str, default="deploy/model", help="Dossier de sortie.")
@click.option("--seed", type=int, default=None, help="Seed de reproductibilité (défaut : config).")
def main(output: str, seed: int | None) -> None:
    """Génère les données, entraîne le modèle officiel, exporte un artefact autonome."""
    overrides = [f"data.seed={seed}"] if seed is not None else []
    cfg = load_config(overrides)

    datasets = generate_datasets(cfg.data)
    features = build_features(datasets.customers, datasets.loans, datasets.savings, cfg.features)
    model, metrics = train_plain(features, cfg.model, cfg.features)

    out_dir = Path(output)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(out_dir),
        serialization_format="skops",
        pip_requirements=["xgboost>=2.0", "scikit-learn>=1.4", "numpy>=1.26", "pandas>=2.1"],
        skops_trusted_types=_SKOPS_TRUSTED_TYPES,
    )
    logger.info("cli.export_model.done", output=str(out_dir.resolve()), roc_auc=metrics["roc_auc"])
    print(f"Modèle exporté dans {out_dir.resolve()} (ROC-AUC={metrics['roc_auc']:.4f})")


if __name__ == "__main__":
    main()
