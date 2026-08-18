"""Chargement de configuration Hydra avec surcharge par fichier YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from cif_credit.config.schema import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONF_DIR = PROJECT_ROOT / "conf"


def load_config(overrides: list[str] | None = None) -> AppConfig:
    """Charge la configuration Hydra + fichiers YAML, puis les surcharges CLI.

    Args:
        overrides: surcharges au format Hydra (ex: ["data.default_rate=0.12"]).

    Returns:
        Config validée.
    """
    base = OmegaConf.structured(AppConfig())
    conf_dir = Path(CONF_DIR)
    if conf_dir.exists():
        for yaml_file in sorted(conf_dir.glob("**/*.yaml")):
            with yaml_file.open("r", encoding="utf-8") as fh:
                data: dict[str, Any] = yaml.safe_load(fh) or {}
            base = OmegaConf.merge(base, data)

    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        base = OmegaConf.merge(base, cli)

    OmegaConf.resolve(base)
    return OmegaConf.to_object(base)  # type: ignore[return-value]
