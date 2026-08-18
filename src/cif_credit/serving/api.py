"""API de scoring FastAPI — modèle servi depuis le registry MLflow, décision via le DecisionEngine.

Le modèle est chargé **au démarrage** (lifespan), pas à l'import, pour rester testable et
suivre le pattern de déploiement des cabinets : injection de dépendances explicite.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from cif_credit.config.schema import FeatureConfig
from cif_credit.decision.engine import DecisionEngine, DecisionPolicy
from cif_credit.models.train import feature_columns
from cif_credit.monitoring.metrics import record_score
from cif_credit.serving.schemas import HealthResponse, ScoreRequest, ScoreResponse
from cif_credit.utils.logging import get_logger

logger = get_logger(__name__)

_START_TIME = time.time()


def _risk_class(probability: float) -> str:
    if probability < 0.1:
        return "faible"
    if probability < 0.25:
        return "moyen"
    if probability < 0.5:
        return "elevé"
    return "critique"


class ModelContainer:
    """Conteneur du modèle chargé au runtime (injecté dans les handlers)."""

    def __init__(self, model: Any | None = None, cols: list[str] | None = None) -> None:
        self.model = model
        self.cols = cols or feature_columns(FeatureConfig())


def create_app(
    model_uri: str | None = None,
    model: Any | None = None,
    policy: DecisionPolicy | None = None,
    feature_columns_list: list[str] | None = None,
) -> FastAPI:
    """Fabrique l'application FastAPI.

    Args:
        model_uri: URI MLflow du modèle (chargé au démarrage si ``model`` est absent).
        model: modèle sklearn directement injecté (pour les tests).
        policy: politique de décision.
        feature_columns_list: liste ordonnée des features.
    """
    uri = model_uri or "models:/cif_credit_official/latest"
    if model is None and os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = ModelContainer(model=model, cols=feature_columns_list)
        if model is None:
            logger.info("serving.load_model", model_uri=uri)
            container.model = mlflow.sklearn.load_model(uri)
        app.state.container = container
        app.state.engine = DecisionEngine(model=container.model, policy=policy or DecisionPolicy())
        yield

    app = FastAPI(
        title="CIF Credit Intelligence API",
        version="1.0.0",
        description="Service de scoring de crédit — CIF Digital Platform §M07. "
        "Les décisions ne sont pas contractuelles : human-in-the-loop obligatoire pour REVUE_HUMAINE.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_version=uri,
            uptime_seconds=round(time.time() - _START_TIME, 2),
        )

    @app.get("/metrics")
    def metrics() -> JSONResponse:
        return JSONResponse(content=generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/score", response_model=ScoreResponse, response_model_exclude_none=True)
    async def score(request: ScoreRequest) -> ScoreResponse:
        container: ModelContainer = app.state.container
        engine: DecisionEngine = app.state.engine
        if container.model is None:
            raise HTTPException(status_code=503, detail="Modèle non chargé")

        start = time.perf_counter()
        provided = set(request.features)
        missing = set(container.cols) - provided
        if missing:
            raise HTTPException(status_code=422, detail=f"Features manquantes : {sorted(missing)}")

        try:
            feat_df = pd.DataFrame([request.features])[container.cols].astype(float)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=f"Feature inconnue : {exc}") from exc

        proba = float(container.model.predict_proba(feat_df)[0, 1])
        outcome = engine.decide(
            probability=proba,
            n_past_loans=request.n_past_loans,
            customer_id=request.customer_id,
        )

        duration = time.perf_counter() - start
        record_score(model_version="latest", probability=proba, duration_seconds=duration)

        return ScoreResponse(
            model_version=uri,
            probability=proba,
            score=outcome.score,
            risk_class=_risk_class(proba),
            decision=outcome.decision.value,
            confidence=outcome.confidence,
            factors={k: round(v, 6) for k, v in outcome.factors.items()},
            policy_hit=outcome.policy_hit,
        )

    return app


def main() -> None:
    """Point d'entrée CLI : uvicorn cif_credit.serving.api:app."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
