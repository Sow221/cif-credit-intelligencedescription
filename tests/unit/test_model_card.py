"""Tests de la Model Card et du registry MLflow (Semaine 4)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from config.schema import FeatureConfig
from models.model_card import ModelCard, ModelCardMetadata, ModelCardRequirementError, build_model_card
from models.registry import find_official_artifact, register_from_joblib, version_exists
from models.train import feature_columns

FEATURES = feature_columns(FeatureConfig())


def _tiny_model() -> XGBClassifier:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    y = (rng.random(100) < 0.2).astype(int)
    model = XGBClassifier(n_estimators=5, max_depth=2)
    model.fit(X, y)
    return model


class TestModelCard:
    def test_metadata_missing_required_field_raises(self):
        with pytest.raises(ModelCardRequirementError):
            ModelCard(
                ModelCardMetadata(
                    model_name="cif_credit_official",
                    version="1.0.0",
                    model_type="classification_binaire",
                    algorithm="xgboost",
                    owner="",
                    contact="",
                    intended_use="",
                    target_population="",
                    decision_thresholds={},
                    training_metrics={},
                )
            )

    def test_build_and_export_json_and_markdown(self, tmp_path):
        card = build_model_card(
            model_name="cif_credit_official",
            version="1.0.0",
            algorithm="xgboost + isotonic",
            owner="TELQAN",
            contact="c@c",
            intended_use="scoring crédit",
            target_population="clients CIF",
            decision_thresholds={"approve": 0.1, "review": 0.25, "hard_reject": 0.5},
            training_metrics={"roc_auc": 0.83, "ece": 0.02},
            features=FEATURES,
        )
        json_path = tmp_path / "card.json"
        md_path = tmp_path / "card.md"
        card.to_json(json_path)
        card.to_markdown(md_path)

        assert json_path.exists()
        assert md_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["name_version"] == "cif_credit_official:1.0.0"
        assert data["training_metrics"]["roc_auc"] == 0.83
        text = md_path.read_text(encoding="utf-8")
        assert "Model Card" in text
        assert "0.1000" in text

    def test_monitoring_thresholds_cabinet_defaults(self):
        card = build_model_card(
            model_name="m",
            version="1",
            algorithm="a",
            owner="o",
            contact="c",
            intended_use="u",
            target_population="p",
            decision_thresholds={"approve": 0.1},
            training_metrics={"roc_auc": 0.8},
        )
        assert card.metadata.monitoring_thresholds == {"roc_auc_min": 0.65, "ece_max": 0.10, "psi_max": 0.25}


class TestRegistry:
    def test_find_official_artifact_by_env(self, tmp_path, monkeypatch):
        artifact = tmp_path / "MODEL_OFFICIAL_CALIBRATED.joblib"
        artifact.write_bytes(b"stub")
        monkeypatch.setenv("CIF_MODEL_ARTIFACT", str(tmp_path))
        found = find_official_artifact()
        assert found is not None
        assert found == artifact

    def test_find_official_artifact_missing(self, tmp_path):
        assert find_official_artifact(candidates=[tmp_path / "nope.joblib"]) is None

    def test_register_from_joblib_logs_and_writes_card(self, tmp_path, monkeypatch):
        tracking = tmp_path / "mlruns.db"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tracking}")
        joblib_path = tmp_path / "model.joblib"
        import joblib

        joblib.dump(_tiny_model(), str(joblib_path))

        registered = register_from_joblib(
            joblib_path,
            version="9.9.9",
            features=FEATURES,
            metrics={"roc_auc": 0.8, "ece": 0.03},
            decision_thresholds={"approve": 0.1, "review": 0.25, "hard_reject": 0.5},
            stage="Staging",
            model_card_dir=tmp_path / "cards",
        )
        assert registered.model_name == "cif_credit_official"
        assert registered.version == "9.9.9"
        assert (tmp_path / "cards" / "cif_credit_official_9.9.9.json").exists()
        assert (tmp_path / "cards" / "cif_credit_official_9.9.9.md").exists()
        assert version_exists("cif_credit_official", "9.9.9")
