"""Tests d'intégration de l'audit trail — journalisation atomique des prédictions (Semaine 3).

Valide les exigences du cabinet (retour.txt §3.3) : chaque prédiction enregistrée avec
ID client, PD, confiance, décision, version du modèle, timestamp et features utilisées ;
les événements d'audit tracés (émission de jeton, prédictions). SQLite fichier en test
(CI sans PostgreSQL) ; le schéma PostgreSQL 16 reste couvert par Alembic.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from api.app import create_app
from config.schema import FeatureConfig
from models.train import feature_columns
from services.audit_service import AuditEvent, AuditService

CLIENT_ID = "cif-agent"
CLIENT_SECRET = "change-me-in-production"
TEST_SECRET = "test-jwt-secret"


def _stub_model() -> XGBClassifier:
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


@pytest.fixture
def audit(tmp_path) -> AuditService:
    url = f"sqlite+pysqlite:///{tmp_path / 'audit.db'}"
    return AuditService(url, create_schema=True)


def test_predict_is_journalized_atomically(audit):
    app = create_app(model=_stub_model(), auth_enabled=False, audit_service=audit, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        resp = client.post("/v1/predict", json=_payload())
        assert resp.status_code == 200
        resp2 = client.post("/v1/predict", json=_payload())
        assert resp2.status_code == 200

    assert audit.count_predictions() == 2
    assert audit.count_events() == 2
    events = audit.recent_events()
    assert all(e["event"] == AuditEvent.PREDICTION_REQUESTED.value for e in events)
    assert all(e["status"] == "ok" for e in events)
    assert all(e["customer_id"] is not None for e in events)


def test_prediction_rows_contain_required_fields(audit):
    app = create_app(model=_stub_model(), auth_enabled=False, audit_service=audit, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        body = _payload()
        resp = client.post("/v1/predict", json=body)
        assert resp.status_code == 200

    rows = audit.recent_events()
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == AuditEvent.PREDICTION_REQUESTED.value
    assert row["request_id"]
    assert row["model_version"] == "models:/cif_credit_official@champion"
    assert "decision" in row["payload"]
    assert "probability" in row["payload"]


def test_token_issuance_is_audited(audit):
    app = create_app(model=_stub_model(), auth_enabled=False, audit_service=audit, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        resp = client.post("/v1/auth/token", json={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET})
        assert resp.status_code == 200

    assert audit.count_events() == 1
    event = audit.recent_events()[0]
    assert event["event"] == AuditEvent.TOKEN_ISSUED.value
    assert event["actor"] == CLIENT_ID


def test_failed_login_is_audited(audit):
    app = create_app(model=_stub_model(), auth_enabled=False, audit_service=audit, jwt_secret=TEST_SECRET)
    with TestClient(app) as client:
        resp = client.post("/v1/auth/token", json={"client_id": "x", "client_secret": "y"})
        assert resp.status_code == 401

    assert audit.count_events() == 1
    assert audit.recent_events()[0]["event"] == AuditEvent.LOGIN_FAILURE.value
    assert audit.recent_events()[0]["status"] == "failed"


def test_audit_direct_record_roundtrip(audit):
    audit.record(
        AuditEvent.MODEL_REGISTERED,
        actor="ci",
        model_version="v0.2.0",
        status="ok",
        payload={"metrics": {"roc_auc": 0.83}},
    )
    events = audit.recent_events()
    assert len(events) == 1
    assert events[0]["event"] == AuditEvent.MODEL_REGISTERED.value
    assert events[0]["model_version"] == "v0.2.0"
    assert events[0]["payload"]["metrics"]["roc_auc"] == 0.83


def test_model_version_registry_upsert(audit):
    audit.upsert_model_version(model_name="cif_credit_official", version="1.0.0", stage="Staging")
    audit.upsert_model_version(model_name="cif_credit_official", version="1.0.0", stage="Production")
    assert audit.count_predictions() == 0
    assert audit.count_events() == 0
