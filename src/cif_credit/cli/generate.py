"""CLI : cif-generate — génération et préparation des données."""

from __future__ import annotations

import click

from cif_credit.config import load_config
from cif_credit.data.synthetic import generate_datasets, save_datasets
from cif_credit.features.builder import build_features, save_features
from cif_credit.utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--n-customers", type=int, default=None, help="Nombre de clients (sur un défaut config).")
@click.option("--default-rate", type=float, default=None, help="Taux de défaut cible.")
@click.option("--seed", type=int, default=None, help="Seed de reproductibilité.")
def main(n_customers: int | None, default_rate: float | None, seed: int | None) -> None:
    """Génère les données synthétiques et construit les features."""
    overrides: list[str] = []
    if n_customers is not None:
        overrides.append(f"data.n_customers={n_customers}")
    if default_rate is not None:
        overrides.append(f"data.default_rate={default_rate}")
    if seed is not None:
        overrides.append(f"data.seed={seed}")

    cfg = load_config(overrides)
    datasets = generate_datasets(cfg.data)
    raw_paths = save_datasets(cfg.data, datasets)

    features = build_features(datasets.customers, datasets.loans, datasets.savings, cfg.features)
    feat_path = save_features(features, cfg.features, cfg.data.processed_dir)

    logger.info(
        "cli.generate.done",
        raw=raw_paths,
        features=feat_path,
        n_features=len(features.columns) - 1,
        default_rate=round(float(features[cfg.features.target].mean()), 4),
    )
