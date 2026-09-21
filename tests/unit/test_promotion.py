from __future__ import annotations

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from models.promotion import CHAMPION, PREVIOUS, PromotionGates, current_champion, promote_if_better, rollback

NAME = "cif_test_model"


@pytest.fixture()
def client(tmp_path) -> MlflowClient:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("promotion_test")
    return MlflowClient()


def _register(roc_auc: float, ece: float = 0.02) -> str:
    x = np.random.default_rng(0).normal(size=(20, 2))
    model = LogisticRegression().fit(x, (x[:, 0] > 0).astype(int))
    with mlflow.start_run():
        mlflow.log_metrics({"roc_auc": roc_auc, "ece": ece})
        info = mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=NAME)
    return str(info.registered_model_version)


def test_first_valid_model_becomes_champion(client: MlflowClient) -> None:
    v = _register(0.72)
    decision = promote_if_better(client, NAME, v)
    assert decision.promoted and current_champion(client, NAME) == v


def test_better_candidate_replaces_champion_and_old_becomes_previous(client: MlflowClient) -> None:
    v1 = _register(0.70)
    promote_if_better(client, NAME, v1)
    v2 = _register(0.74)
    assert promote_if_better(client, NAME, v2).promoted
    assert current_champion(client, NAME) == v2
    assert str(client.get_model_version_by_alias(NAME, PREVIOUS).version) == v1


def test_worse_candidate_is_refused_and_champion_unchanged(client: MlflowClient) -> None:
    v1 = _register(0.74)
    promote_if_better(client, NAME, v1)
    v2 = _register(0.70)
    d = promote_if_better(client, NAME, v2)
    assert not d.promoted and "ne bat pas" in d.reason
    assert current_champion(client, NAME) == v1


def test_poorly_calibrated_or_weak_candidates_are_refused(client: MlflowClient) -> None:
    assert not promote_if_better(client, NAME, _register(0.80, ece=0.25)).promoted
    assert not promote_if_better(client, NAME, _register(0.55)).promoted
    assert current_champion(client, NAME) is None


def test_min_gain_is_enforced(client: MlflowClient) -> None:
    v1 = _register(0.70)
    promote_if_better(client, NAME, v1)
    v2 = _register(0.703)
    assert not promote_if_better(client, NAME, v2, PromotionGates(min_gain=0.01)).promoted


def test_rollback_restores_previous(client: MlflowClient) -> None:
    v1 = _register(0.70)
    promote_if_better(client, NAME, v1)
    v2 = _register(0.75)
    promote_if_better(client, NAME, v2)
    assert rollback(client, NAME) == v1
    assert str(client.get_model_version_by_alias(NAME, CHAMPION).version) == v1


def test_rollback_without_previous_fails(client: MlflowClient) -> None:
    promote_if_better(client, NAME, _register(0.70))
    with pytest.raises(ValueError, match="rollback impossible"):
        rollback(client, NAME)
