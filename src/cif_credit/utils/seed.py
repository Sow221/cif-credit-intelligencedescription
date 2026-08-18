"""Utilitaires de reproductibilité (seeds globaux)."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    """Fixe les seeds Python, NumPy et éventuellement scikit-learn pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        from sklearn.utils import check_random_state  # noqa: F401

        # s'assure que sklearn utilise bien le générateur numpy
        np.random.seed(seed)
    except ImportError:
        pass


def safe_str(value: Any) -> str:
    """Convertit une valeur en chaîne robuste (NaN, None, bool...)."""
    if value is None:
        return "None"
    if isinstance(value, float) and np.isnan(value):
        return "NaN"
    return str(value)
