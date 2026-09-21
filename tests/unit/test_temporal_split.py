"""Tests de la discipline de split TEMPOREL (protocole CIF).

Verrouille que l'entraînement n'utilise JAMAIS de split aléatoire : la coupure
suit l'ordre temporel réel (``application_date``, générée par une jointure
point-in-time — voir ``data/synthetic.py``) et est strictement reproductible.
Un split aléatoire ici invaliderait la validité temporelle de toutes les
métriques.
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
    """L'ordre temporel réel doit être respecté : train avant test."""
    # Volontairement mélangé (ordre customer_id, PAS l'ordre temporel) : le split doit
    # retrier lui-même sur application_date, jamais faire confiance à l'ordre d'entrée.
    df = _features().sort_values("customer_id").reset_index(drop=True)
    cfg = ModelConfig()
    X_tr, X_te, _, _ = temporal_split(df, cfg, "is_default")
    assert X_tr.index.max() < X_te.index.min()


def test_temporal_split_sorts_by_application_date_not_input_order() -> None:
    """Le split doit suivre application_date même si le DataFrame arrive dans un tout
    autre ordre (ex. trié par customer_id, qui n'est pas corrélé à la date de demande)."""
    df = _features()
    shuffled = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    cfg = ModelConfig()
    X_tr, X_te, y_tr, y_te = temporal_split(shuffled, cfg, "is_default", feature_cfg=FeatureConfig())
    X_tr2, X_te2, y_tr2, y_te2 = temporal_split(df, cfg, "is_default", feature_cfg=FeatureConfig())
    assert X_tr.reset_index(drop=True).equals(X_tr2.reset_index(drop=True))
    assert X_te.reset_index(drop=True).equals(X_te2.reset_index(drop=True))
    assert np.array_equal(y_tr, y_tr2)
    assert np.array_equal(y_te, y_te2)
