"""Tests de la configuration et de l'API (sans déploiement MLflow)."""


import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from cif_credit.config import load_config
from cif_credit.config.schema import FeatureConfig
from cif_credit.models.train import feature_columns
from cif_credit.serving.api import create_app


def test_load_config_defaults():
    cfg = load_config()
    assert cfg.data.n_customers == 10_000
    assert cfg.data.default_rate == pytest.approx(0.1183)
    assert cfg.model.algorithm == "xgboost"
    assert cfg.evaluation.go_roc_auc == 0.65


def test_load_config_override():
    cfg = load_config(["data.n_customers=123", "decision.approve_threshold=0.2"])
    assert cfg.data.n_customers == 123
    assert cfg.decision.approve_threshold == 0.2


def test_feature_columns_length():
    cfg = FeatureConfig()
    assert len(feature_columns(cfg)) == 25


def test_api_health_and_score():
    app = create_app(model=_stub_model())
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        cols = feature_columns(FeatureConfig())
        payload = {
            "customer_id": 1,
            "features": {c: 0.5 for c in cols},
            "n_past_loans": 2,
        }
        resp = client.post("/v1/score", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["decision"] in {"APPROBATION", "AJUSTEMENT", "REVUE_HUMAINE", "REFUS"}


def test_api_rejects_missing_features():
    app = create_app(model=_stub_model())
    with TestClient(app) as client:
        resp = client.post("/v1/score", json={"customer_id": 1, "features": {}, "n_past_loans": 0})
        assert resp.status_code == 422


def _stub_model() -> XGBClassifier:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    cols = feature_columns(FeatureConfig())
    X = pd.DataFrame(rng.normal(size=(200, len(cols))), columns=cols)
    y = (rng.random(200) < 0.2).astype(int)
    model = XGBClassifier(n_estimators=10, max_depth=3)
    model.fit(X, y)
    return model
