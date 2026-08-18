"""API de scoring FastAPI — modèle servi depuis le registry MLflow, décision via le DecisionEngine.

Assemblage exigé par le cabinet (Semaine 2) : ``app.py`` (fabrique), ``routes.py`` (endpoints),
``schemas.py`` (Pydantic strict), ``middleware.py`` (request ID + rate limiting) et JWT.
Le modèle est chargé **au démarrage** (lifespan), pas à l'import, pour rester testable et
suivre le pattern d'injection de dépendances explicite.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from fastapi import FastAPI

from api.middleware import RateLimiter, RequestIDMiddleware
from api.routes import router_plain, router_v1
from config.schema import FeatureConfig
from config.settings import get_settings
from models.train import feature_columns
from services.decision_engine import DecisionEngine, DecisionPolicy
from utils.logging import get_logger

logger = get_logger(__name__)

_START_TIME = time.time()


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
    *,
    auth_enabled: bool = True,
    jwt_secret: str | None = None,
    jwt_ttl_minutes: int = 60,
    rate_limiter: RateLimiter | None = None,
) -> FastAPI:
    """Fabrique l'application FastAPI.

    Args:
        model_uri: URI MLflow du modèle (chargé au démarrage si ``model`` est absent).
        model: modèle sklearn directement injecté (pour les tests).
        policy: politique de décision.
        feature_columns_list: liste ordonnée des features.
        auth_enabled: active/désactive l'authentification JWT (tests).
        jwt_secret: secret HS256 (défaut : settings Pydantic).
        jwt_ttl_minutes: durée de validité des jetons.
        rate_limiter: instance RateLimiter (tests) — défaut : settings.
    """
    settings = get_settings()
    uri = model_uri or "models:/cif_credit_official/latest"
    if model is None and os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    secret = jwt_secret or settings.jwt_secret.get_secret_value()

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
        swagger_ui_parameters={"persistAuthorization": True},
    )

    app.state.auth_enabled = auth_enabled
    app.state.jwt_secret = secret
    app.state.jwt_ttl_minutes = jwt_ttl_minutes
    app.state.rate_limiter = rate_limiter or RateLimiter(settings.api.request_limits_rate_per_minute)
    app.state.start_time = _START_TIME
    app.state.model_version = uri

    app.add_middleware(RequestIDMiddleware)
    app.include_router(router_v1)
    app.include_router(router_plain)

    return app


def main() -> None:
    """Point d'entrée CLI : uvicorn api.app:create_app."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
