"""API de scoring FastAPI — modèle servi depuis le registry MLflow, décision via le DecisionEngine.

Assemblage exigé par le cabinet (Semaine 2) : ``app.py`` (fabrique), ``routes.py`` (endpoints),
``schemas.py`` (Pydantic strict), ``middleware.py`` (request ID + rate limiting) et JWT.
Le modèle est chargé **au démarrage** (lifespan), pas à l'import, pour rester testable et
suivre le pattern d'injection de dépendances explicite.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from fastapi import FastAPI

from api.middleware import MetricsMiddleware, RateLimiter, RequestIDMiddleware
from api.routes import router_plain, router_v1
from api.security import assert_production_secrets
from config.schema import FeatureConfig
from config.settings import get_settings
from models.train import feature_columns
from monitoring.drift_monitor import DRIFT_COMPUTE_INTERVAL_SECONDS, DriftMonitor, load_reference
from services.audit_service import AuditService
from services.decision_engine import DecisionEngine, DecisionPolicy
from services.predictor import Predictor
from utils.logging import get_logger

logger = get_logger(__name__)

_START_TIME = time.time()


class ModelContainer:
    """Conteneur du modèle chargé au runtime (injecté dans les handlers)."""

    def __init__(self, model: Any | None = None, cols: list[str] | None = None) -> None:
        self.model = model
        self.cols = cols or feature_columns(FeatureConfig())


async def _drift_loop(monitor: DriftMonitor) -> None:
    """Boucle d'arrière-plan : recalcule le drift périodiquement."""
    while True:
        await asyncio.sleep(DRIFT_COMPUTE_INTERVAL_SECONDS)
        try:
            monitor.compute()
        except Exception:  # pragma: no cover
            logger.exception("monitoring.drift.loop.error")


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
    audit_service: AuditService | None = None,
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
        audit_service: service d'audit (Semaine 3). Si ``None`` et que l'URL de la base
            est fournie via ``CIF_DATABASE__URL``, il est construit automatiquement.
    """
    settings = get_settings()
    # Ordre de résolution : argument > CIF_MODEL_URI / MODEL_URI (images déployées, modèle
    # exporté sur disque) > registry MLflow (stack locale).
    uri = (
        model_uri
        or os.environ.get("CIF_MODEL_URI")
        or os.environ.get("MODEL_URI")
        or "models:/cif_credit_official@champion"
    )
    assert_production_secrets(jwt_secret or os.environ.get("CIF_JWT_SECRET"))
    if model is None and os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    secret = jwt_secret or settings.jwt_secret.get_secret_value()
    if audit_service is None and os.environ.get("CIF_DATABASE__URL"):
        audit_service = AuditService(url=os.environ["CIF_DATABASE__URL"])

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = ModelContainer(model=model, cols=feature_columns_list)
        if model is None:
            logger.info("serving.load_model", model_uri=uri)
            container.model = mlflow.sklearn.load_model(uri)
        app.state.container = container
        app.state.engine = DecisionEngine(model=container.model, policy=policy or DecisionPolicy())
        app.state.predictor = Predictor(
            app.state.engine,
            model_version=uri,
            audit=audit_service,
        )

        # Monitoring de drift temps réel (Evidently) — désactivé proprement si la
        # baseline ne peut être construite (ex : dépendance manquante).
        cols = feature_columns(FeatureConfig())
        try:
            reference = load_reference(cols)
            app.state.drift_monitor = DriftMonitor(reference, cols)
        except Exception as exc:  # pragma: no cover
            logger.warning("monitoring.drift.disabled", error=str(exc))
            app.state.drift_monitor = None
        drift_task = None
        if app.state.drift_monitor is not None:
            drift_task = asyncio.create_task(_drift_loop(app.state.drift_monitor))

        try:
            yield
        finally:
            if drift_task is not None:
                drift_task.cancel()

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
    app.state.audit_service = audit_service
    app.state.start_time = _START_TIME
    app.state.model_version = uri

    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.include_router(router_v1)
    app.include_router(router_plain)

    return app


def main() -> None:
    """Point d'entrée CLI : uvicorn api.app:create_app."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
