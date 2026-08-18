"""Tests d'intégration de l'API — JWT, rate limiting, endpoints versionnés (Semaine 2).

Couvre les exigences du cabinet : authentification JWT, rate limiting par client,
validation stricte extra="forbid", endpoints /v1/predict et /v1/health.
"""

from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from api.app import create_app
from api.middleware import RateLimiter
from api.security import check_credentials
from config.schema import FeatureConfig
from models.train import feature_columns

TEST_SECRET = "test-jwt-secret"
CLIENT_ID = "cif-agent"
CLIENT_SECRET = "change-me-in-production"


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


def _payload() -> dict:
    cols = feature_columns(FeatureConfig())
    return {"customer_id": 42, "features": {c: 0.5 for c in cols}, "n_past_loans": 2}


def _get_token(client: TestClient) -> str:
    resp = client.post("/v1/auth/token", json={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def test_health_endpoint_versioned_and_plain():
    app = create_app(model=_stub_model(), auth_enabled=False, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/health").json()["status"] == "ok"


def test_token_endpoint_returns_jwt():
    app = create_app(model=_stub_model(), auth_enabled=False, jwt_secret=TEST_SECRET, jwt_ttl_minutes=5)
    with TestClient(app) as client:
        token = _get_token(client)
        assert token
        assert len(token.split(".")) == 3


def test_token_endpoint_rejects_bad_credentials():
    app = create_app(model=_stub_model(), auth_enabled=False, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        resp = client.post("/v1/auth/token", json={"client_id": "x", "client_secret": "y"})
        assert resp.status_code == 401


def test_predict_requires_auth_when_enabled():
    app = create_app(model=_stub_model(), auth_enabled=True, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        resp = client.post("/v1/predict", json=_payload())
        assert resp.status_code == 401


def test_predict_rejects_invalid_token():
    app = create_app(model=_stub_model(), auth_enabled=True, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/predict",
            json=_payload(),
            headers={"Authorization": "Bearer not-a-valid-token"},
        )
        assert resp.status_code == 401


def test_predict_accepts_valid_token():
    app = create_app(model=_stub_model(), auth_enabled=True, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        token = _get_token(client)
        resp = client.post("/v1/predict", json=_payload(), headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["decision"] in {"APPROBATION", "AJUSTEMENT", "REVUE_HUMAINE", "REFUS"}


def test_request_id_header_is_returned():
    app = create_app(model=_stub_model(), auth_enabled=False, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        resp = client.get("/v1/health")
        assert resp.headers.get("X-Request-ID")


def test_rate_limiting_per_client():
    app = create_app(
        model=_stub_model(),
        auth_enabled=False,
        jwt_secret=TEST_SECRET,
        rate_limiter=RateLimiter(rate_per_minute=3),
    )
    with TestClient(app) as client:
        for _ in range(3):
            assert client.post("/v1/predict", json=_payload()).status_code == 200
        resp = client.post("/v1/predict", json=_payload())
        assert resp.status_code == 429


def test_check_credentials_uses_env(monkeypatch):
    monkeypatch.setenv("CIF_API_CLIENT_ID", "agent-1")
    monkeypatch.setenv("CIF_API_CLIENT_SECRET", "secret-1")
    assert check_credentials("agent-1", "secret-1")
    assert not check_credentials("agent-1", "wrong")
