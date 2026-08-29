"""Tests du feature engineering et du modèle."""

import pandas as pd

from config.schema import DataConfig, FeatureConfig, ModelConfig
from data.synthetic import generate_datasets
from features.builder import build_features
from models.train import feature_columns, train_plain


def _fixture() -> tuple[pd.DataFrame, FeatureConfig]:
    cfg = DataConfig(n_customers=400, seed=5)
    ds = generate_datasets(cfg)
    feat_cfg = FeatureConfig()
    features = build_features(ds.customers, ds.loans, ds.savings, feat_cfg)
    return features, feat_cfg


def test_build_features_shape():
    features, feat_cfg = _fixture()
    cols = feature_columns(feat_cfg)
    assert len(cols) == 25
    assert set(cols).issubset(features.columns)
    assert feat_cfg.target in features.columns
    assert len(features) == 400


def test_thin_file_imputation():
    features, _ = _fixture()
    thin = features[features["n_past_loans"] == 0]
    if len(thin) > 0:
        # Les thin-file ont les agrégats de prêt remplis à neutre (0)
        assert (thin["n_loans"] == 0).all()
        assert (thin["max_historical_dpd"] == 0).all()
        assert (thin["avg_repayment_regularity"] == 0).all()


def test_train_plain_runs():
    features, feat_cfg = _fixture()
    model_cfg = ModelConfig(n_trials=5)
    model, metrics = train_plain(features, model_cfg, feat_cfg)
    assert model is not None
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0.5
