"""Registry MLflow — enregistrement des modèles, stages, model cards (Semaine 4).

Objectif cabinet : « Modèle enregistré dans MLflow — mlflow ui montre le modèle ».
Ce module centralise l'enregistrement (depuis artifact joblib du prototype pilote ou depuis
un re-entraînement reproductible), la transition de stage (Staging → Production → Archived)
et la génération de la Model Card associée.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import mlflow
import pandas as pd

from models.model_card import build_model_card
from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "cif_credit_official"
_OFFICIAL_FILENAME = "MODEL_OFFICIAL_CALIBRATED.joblib"


@dataclass
class RegisteredModel:
    """Résultat d'un enregistrement registry."""

    model_name: str
    version: str
    run_id: str
    stage: str = "Staging"


def find_official_artifact(candidates: list[str | Path] | None = None) -> Path | None:
    """Localise le modèle officiel du pilote (env ``CIF_MODEL_ARTIFACT`` prioritaire)."""
    env_path = os.environ.get("CIF_MODEL_ARTIFACT")
    env_candidates: list[str | Path] = [Path(env_path) / _OFFICIAL_FILENAME] if env_path else []
    search_paths = candidates or env_candidates
    for path in search_paths:
        p = Path(path)
        if p.is_file():
            return p
        if p.is_dir():
            candidate = p / _OFFICIAL_FILENAME
            if candidate.is_file():
                return candidate
    return None


def load_joblib_model(path: str | Path) -> Any:
    """Charge un artifact modèle joblib (pipeline calibré complet)."""
    return joblib.load(str(path))


def _create_model_version(
    model: Any,
    *,
    model_name: str,
    version: str,
    features: list[str],
    metrics: dict[str, float],
    stage: str,
) -> str:
    """Enregistre le modèle dans MLflow, transitionne le stage et retourne le run_id.

    MLflow numérote les versions en entiers (auto-incrément) ; la version sémantique
    (ex. 1.0.0) est stockée en tag ``cif_semantic_version`` sur la version créée.
    """
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    with mlflow.start_run(run_name=f"{model_name}-{version}") as run:
        mlflow.set_tag("cif_model_name", model_name)
        mlflow.set_tag("cif_model_version", version)
        mlflow.log_params({"n_features": len(features)})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=pd.DataFrame([{c: 0.0 for c in features}]),
            registered_model_name=model_name,
            serialization_format="cloudpickle",
            pip_requirements=["xgboost>=2.0", "scikit-learn>=1.4", "numpy>=1.26", "pandas>=2.1"],
        )

        client = mlflow.tracking.MlflowClient()
        versions: list[Any] = client.search_model_versions(f"name='{model_name}'") or []
        latest = sorted(int(v.version) for v in versions)[-1] if versions else None
        if latest is not None:
            client.set_model_version_tag(model_name, str(latest), "cif_semantic_version", version)
            client.transition_model_version_stage(model_name, str(latest), stage)
        return cast(str, run.info.run_id)


def register_from_joblib(
    joblib_path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    version: str,
    features: list[str],
    metrics: dict[str, float],
    decision_thresholds: dict[str, float],
    stage: str = "Staging",
    model_card_dir: str | Path = "data/model_cards",
) -> RegisteredModel:
    """Enregistre le modèle officiel pilote (joblib calibré) dans le registry MLflow."""
    model = load_joblib_model(joblib_path)
    run_id = _create_model_version(
        model,
        model_name=model_name,
        version=version,
        features=features,
        metrics=metrics,
        stage=stage,
    )

    card = build_model_card(
        model_name=model_name,
        version=version,
        algorithm="XGBClassifier + CalibratedClassifierCV(isotonic)",
        owner="TELQAN / CIF Credit Intelligence",
        contact="contact@cif-digital-platform.test",
        intended_use=(
            "Scoring de crédit (estimation de la probabilité de défaut) des clients CIF/DigiCoop-WA+ ; "
            "décision finale sous revue humaine (human-in-the-loop) obligatoire."
        ),
        target_population="Détenteurs de produits CIF/DigiCoop-WA+ (pilot synthétique et réel).",
        decision_thresholds=decision_thresholds,
        training_metrics=metrics,
        features=features,
        limitations=[
            "Modèle entraîné sur données synthétiques pour le pilot.",
            "Pas de décision contractuelle sans revue humaine (REVUE_HUMAINE).",
            "À ré-entraîner sur les données réelles CIF (protocole shadow mode).",
        ],
        fairness_notes=(
            "Audit d'équité à reproduire sur données réelles (disparités par profil non fiabilisées "
            "sur données synthétiques)."
        ),
        data_sources=["Prototype synthétique TELQAN (audit phases A-F)"],
    )
    card_dir = Path(model_card_dir)
    card_dir.mkdir(parents=True, exist_ok=True)
    card.to_json(card_dir / f"{model_name}_{version}.json")
    card.to_markdown(card_dir / f"{model_name}_{version}.md")

    logger.info("models.registry.registered", model_name=model_name, version=version, run_id=run_id)
    return RegisteredModel(model_name=model_name, version=version, run_id=run_id, stage=stage)


def version_exists(model_name: str, version: str) -> bool:
    """True si une version du modèle porte déjà la version sémantique ``version``."""
    client = mlflow.tracking.MlflowClient()
    try:
        versions: list[Any] = client.search_model_versions(f"name='{model_name}'") or []
    except mlflow.exceptions.RestException:
        return False
    return any(v.tags.get("cif_semantic_version") == version for v in versions)
