"""Tests de la discipline de split TEMPOREL (protocole CIF).

Verrouille que l'entraînement n'utilise JAMAIS de split aléatoire : la coupure
suit l'ordre temporel (proxy : ``customer_id`` croissant sur données
synthétiques) et est strictement reproductible. Un split aléatoire ici
invaliderait la validité temporelle de toutes les métriques.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import DataConfig, FeatureConfig, ModelConfig
from data.synthetic import generate_datasets
from features.builder import build_features
from models.train import temporal_split


def _features() -> pd.DataFrame:
    ds = generate_datasets(DataConfig(n_customers=120, seed=5))
    return build_features(ds.customers, ds.loans, ds.savings, FeatureConfig())


def test_temporal_split_is_deterministic() -> None:
    df = _features()
    cfg = ModelConfig()
    a = temporal_split(df, cfg, "is_default")
    b = temporal_split(df, cfg, "is_default")
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_temporal_split_split_is_disjoint_exhaustive() -> None:
    df = _features()
    cfg = ModelConfig()
    X_tr, X_te, _, _ = temporal_split(df, cfg, "is_default")
    train_ids = set(X_tr.index)
    test_ids = set(X_te.index)
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == len(df)


def test_temporal_split_preserves_order() -> None:
    """L'ordre (proxy temporel) doit être respecté : train avant test."""
    df = _features().sort_values("customer_id").reset_index(drop=True)
    cfg = ModelConfig()
    X_tr, X_te, _, _ = temporal_split(df, cfg, "is_default")
    assert X_tr.index.max() < X_te.index.min()
