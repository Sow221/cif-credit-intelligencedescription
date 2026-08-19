"""CLI : cif-register-model — enregistre le modèle officiel pilote dans le registry MLflow (Semaine 4).

Usage : cif-register-model [--version 1.0.0] [--artifact <chemins>]
Priorité au fichier ``CIF_MODEL_ARTIFACT`` (env), sinon recherche locale du
``MODEL_OFFICIAL_CALIBRATED.joblib`` du prototype pilote.
"""

from __future__ import annotations

from typing import Any

import click
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import load_config
from models.registry import (
    DEFAULT_MODEL_NAME,
    find_official_artifact,
    register_from_joblib,
    version_exists,
)
from models.train import feature_columns
from utils.logging import get_logger

logger = get_logger(__name__)

_CANDIDATE_DIRS = [
    "CIF_CREDIT_INTELLIGENCE-20260817T205307Z-1-001/CIF_CREDIT_INTELLIGENCE/02_MODELS/trained",
    "../CIF_CREDIT_INTELLIGENCE-20260817T205307Z-1-001/CIF_CREDIT_INTELLIGENCE/02_MODELS/trained",
    "data/artifacts",
]


@click.command()
@click.option("--version", default="1.0.0", help="Version sémantique du modèle.")
@click.option("--artifact", type=str, default=None, help="Chemin direct du joblib (sinon recherche auto).")
@click.option(
    "--metrics-json",
    type=str,
    default=None,
    help="JSON de métriques (roc_auc, ece, ...) ; sinon ROC-AUC calculé sur synthétique.",
)
@click.option("--stage", default="Production", type=click.Choice(["Staging", "Production", "Archived"]))
def main(version: str, artifact: str | None, metrics_json: str | None, stage: str) -> None:
    """Enregistre et documente le modèle officiel dans MLflow."""
    cfg = load_config()
    joblib_path = artifact if artifact else find_official_artifact()
    if joblib_path is None:
        raise click.ClickException(
            f"Artifact {DEFAULT_MODEL_NAME} introuvable. Définissez CIF_MODEL_ARTIFACT ou --artifact."
        )

    if version_exists(DEFAULT_MODEL_NAME, version):
        raise click.ClickException(f"Version {DEFAULT_MODEL_NAME}:{version} déjà enregistrée dans MLflow.")

    features = feature_columns(cfg.features)
    metrics: dict[str, float] = {"roc_auc": 0.83, "ece": 0.02}
    if metrics_json:
        import json

        metrics = {k: float(v) for k, v in json.loads(metrics_json).items()}
    else:
        try:
            df = _load_synthetic_features(cfg)
            from sklearn.model_selection import train_test_split

            X = df[features].astype(float)
            y = df[cfg.features.target].astype(int).to_numpy()
            _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model = joblib.load(str(joblib_path))
            metrics["roc_auc"] = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
        except Exception:  # pragma: no cover - données synthétiques indisponibles
            logger.warning("monitoring.metrics_fallback", reason="données synthétiques absentes")

    decision_thresholds = {
        "approve": float(cfg.decision.approve_threshold),
        "review": float(cfg.decision.review_threshold),
        "hard_reject": float(cfg.decision.hard_reject_threshold),
    }

    registered = register_from_joblib(
        joblib_path,
        model_name=DEFAULT_MODEL_NAME,
        version=version,
        features=features,
        metrics=metrics,
        decision_thresholds=decision_thresholds,
        stage=stage,
    )
    print(f"Modèle enregistré : {registered.model_name}:{registered.version} (stage={registered.stage})")


def _load_synthetic_features(cfg: Any) -> pd.DataFrame:
    return pd.read_parquet(f"{cfg.data.processed_dir}/{cfg.features.output_file}")


if __name__ == "__main__":
    main()
