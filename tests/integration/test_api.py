"""Tests d'intégration de l'API — JWT, rate limiting, endpoints versionnés (Semaine 2).

Couvre les exigences du cabinet : authentification JWT, rate limiting par client,
validation stricte extra="forbid", endpoints /v1/predict et /v1/health.
"""

import pytest
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


def test_predict_rejects_forbidden_feature():
    """Garde anti-leakage : une feature de fuite (p_default_true) → 422."""
    app = create_app(model=_stub_model(), auth_enabled=False, jwt_secret=TEST_SECRET)
    payload = _payload()
    payload["features"]["p_default_true"] = 0.5
    with TestClient(app) as client:
        resp = client.post("/v1/predict", json=payload)
        assert resp.status_code == 422


def test_metrics_endpoint_is_prometheus_text():
    app = create_app(model=_stub_model(), auth_enabled=False, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "cif_http_requests_total" in r.text


def test_production_refuses_default_secrets(monkeypatch):
    monkeypatch.setenv("CIF_ENV", "production")
    monkeypatch.delenv("CIF_API_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("CIF_JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="production"):
        create_app(model=_stub_model(), auth_enabled=False)


def test_model_uri_read_from_env(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "/app/model")
    app = create_app(model=_stub_model(), auth_enabled=False, jwt_secret=TEST_SECRET)
    assert app.state.model_version == "/app/model"


def test_lending_club_score_returns_503_when_model_not_loaded():
    app = create_app(model=_stub_model(), auth_enabled=False, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        app.state.lending_club_predictor = None
        r = client.post("/v1/lending-club/score", json={"features": {}})
    assert r.status_code == 503


def test_lending_club_score_end_to_end_with_stub_champion():
    import numpy as np
    import pandas as pd

    from data.lending_club import temporal_partition
    from features.lending_club import FEATURES
    from models.scoring import fit_logistic
    from services.lending_club_predictor import LendingClubPredictor

    rng = np.random.default_rng(0)
    from config.schema import LendingClubConfig

    cfg = LendingClubConfig()
    n = 400
    df = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(2000, 30000, n),
            "installment": rng.uniform(50, 900, n),
            "emp_length": rng.uniform(0, 10, n),
            "annual_inc_log": rng.normal(11, 0.5, n),
            "dti": rng.uniform(0, 40, n),
            "delinq_2yrs": rng.poisson(0.2, n).astype(float),
            "credit_history_years": rng.uniform(1, 30, n),
            "fico_mean": rng.normal(700, 30, n),
            "inq_last_6mths": rng.poisson(0.5, n).astype(float),
            "open_acc": rng.integers(2, 20, n).astype(float),
            "pub_rec": rng.poisson(0.1, n).astype(float),
            "revol_bal_log": rng.normal(8, 1, n),
            "revol_util": rng.uniform(0, 100, n),
            "total_acc": rng.integers(5, 40, n).astype(float),
            "mort_acc": rng.poisson(1, n).astype(float),
            "pub_rec_bankruptcies": rng.poisson(0.05, n).astype(float),
            "loan_to_income": rng.uniform(0.05, 0.8, n),
            "installment_to_income": rng.uniform(0.01, 0.3, n),
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
            "verification_status": rng.choice(["Verified", "Not Verified"], n),
            "purpose": rng.choice(["debt_consolidation", "credit_card", "car"], n),
            "application_type": "Individual",
            "issue_d": pd.date_range("2012-01-01", periods=n, freq="7D"),
            "is_default": (rng.random(n) < 0.15).astype(int),
        }
    )
    train, _, _ = temporal_partition(df, cfg, date_col="issue_d")
    model = fit_logistic(train, train["is_default"].to_numpy()).calibrate(
        train.head(50), train["is_default"].head(50).to_numpy()
    )
    stub_predictor = LendingClubPredictor(model, "test:/lending_club_champion", threshold=0.15)

    app = create_app(model=_stub_model(), auth_enabled=False, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        app.state.lending_club_predictor = stub_predictor
        payload = {"features": train[list(FEATURES)].iloc[0].to_dict()}
        r = client.post("/v1/lending-club/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["decision"] in {"APPROBATION", "REVUE_HUMAINE", "REFUS"}
    assert body["model_version"] == "test:/lending_club_champion"
    assert set(body["factors"]) <= set(FEATURES)


def test_lending_club_score_rejects_missing_features():
    app = create_app(model=_stub_model(), auth_enabled=False, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        r = client.post("/v1/lending-club/score", json={"features": {"loan_amnt": 1000.0}})
    assert r.status_code in {422, 503}
