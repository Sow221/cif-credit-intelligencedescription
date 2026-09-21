"""CLI : cif-ingest-lc — raw → interim (validé) → features (point-in-time)."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from config import load_config
from data.lending_club import build_interim
from features.lending_club import build_features
from utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--nrows", type=int, default=None, help="Limite de lignes lues (essais rapides).")
def main(nrows: int | None) -> None:
    """Nettoie et valide le jeu Lending Club puis construit les features."""
    cfg = load_config().lending_club
    interim = pd.read_parquet(build_interim(cfg, nrows=nrows))
    features = build_features(interim)
    out = Path(cfg.processed_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    logger.info("cli.ingest_lc.done", rows=len(features), path=str(out))
