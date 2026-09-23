"""Routes de l'API — /v1/health, /v1/predict, /v1/auth/token, /v1/score, /metrics.

Exigences cabinet (Semaine 2) : endpoints versionnés, validation stricte ``extra="forbid"``
(schémas), rate limiting par client, authentification JWT. L'état (modèle, engine, config)
est injecté par ``app.py`` via ``app.state`` (pattern d'injection de dépendances).
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.middleware import RateLimiterProtocol
from api.schemas import (
    HealthResponse,
    LendingClubScoreRequest,
    LendingClubScoreResponse,
    ScoreRequest,
    ScoreResponse,
    TokenRequest,
    TokenResponse,
)
from api.security import check_credentials, create_access_token, verify_token
from features.lending_club import FEATURES as LC_FEATURES
from features.validate import FORBIDDEN_SUBSTRINGS
from monitoring.metrics import record_score
from services.audit_service import AuditEvent, AuditService
from services.lending_club_predictor import LendingClubPredictor
from services.predictor import Predictor
from utils.logging import get_logger

logger = get_logger(__name__)

router_v1 = APIRouter(prefix="/v1")
router_plain = APIRouter()


def _client_identity(request: Request) -> str:
    """Identifie le client via son jeton JWT (ou IP en mode non authentifié)."""
    app = request.app
    if app.state.auth_enabled:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Jeton manquant (Authorization: Bearer <token>)")
        token = auth.removeprefix("Bearer ").strip()
        try:
            payload = verify_token(token, app.state.jwt_secret)
        except Exception as exc:
            raise HTTPException(status_code=401, detail="Jeton invalide ou expiré") from exc
        return payload.sub
    return request.headers.get("X-Client-Id", request.client.host if request.client else "anonymous")


async def _enforce_rate_limit(request: Request, client: str) -> None:
    rate_limiter: RateLimiterProtocol = request.app.state.rate_limiter
    if not await rate_limiter.allow(client):
        raise HTTPException(status_code=429, detail="Rate limit dépassé (requêtes/minute)")


def _audit(request: Request, event: AuditEvent, *, actor: str | None = None, status: str | None = None) -> None:
    service: AuditService | None = request.app.state.audit_service
    if service is not None:
        service.record(event, actor=actor, request_id=request.headers.get("X-Request-ID"), status=status)


@router_plain.get("/health", response_model=HealthResponse, tags=["system"])
@router_v1.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    """Sonde de santé / liveness (accessible sans authentification)."""
    return HealthResponse(
        status="ok",
        model_version=request.app.state.model_version,
        uptime_seconds=round(time.time() - request.app.state.start_time, 2),
    )


@router_plain.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Métriques Prometheus exposées au scraper."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router_v1.post("/auth/token", response_model=TokenResponse, tags=["auth"])
def token(body: TokenRequest, request: Request) -> TokenResponse:
    """Échange des credentials client contre un jeton JWT (Bearer)."""
    if not check_credentials(body.client_id, body.client_secret):
        _audit(request, AuditEvent.LOGIN_FAILURE, actor=body.client_id, status="failed")
        raise HTTPException(status_code=401, detail="Credentials invalides")
    ttl = request.app.state.jwt_ttl_minutes
    access = create_access_token(body.client_id, request.app.state.jwt_secret, ttl_minutes=ttl)
    logger.info("api.auth.token_issued", client=body.client_id, ttl_minutes=ttl)
    _audit(request, AuditEvent.TOKEN_ISSUED, actor=body.client_id, status="ok")
    return TokenResponse(access_token=access, expires_in=ttl * 60)


@router_v1.post("/predict", response_model=ScoreResponse, tags=["scoring"])
async def predict(
    payload: ScoreRequest,
    request: Request,
    client: str = Depends(_client_identity),
) -> ScoreResponse:
    """Scoring d'une demande de crédit + décision (human-in-the-loop)."""
    return await _score_handlers(payload, request, client)


@router_v1.post("/score", response_model=ScoreResponse, tags=["scoring"], include_in_schema=False)
async def score_alias(
    payload: ScoreRequest,
    request: Request,
    client: str = Depends(_client_identity),
) -> ScoreResponse:
    """Alias de compatibilité de /v1/predict."""
    return await _score_handlers(payload, request, client)


async def _score_handlers(payload: ScoreRequest, request: Request, client: str) -> ScoreResponse:
    await _enforce_rate_limit(request, client)
    container: Any = request.app.state.container
    predictor: Predictor = request.app.state.predictor
    if container.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    start = time.perf_counter()
    provided = set(payload.features)

    # Garde anti-leakage : rejette formellement toute feature de fuite (422).
    forbidden_hits = [f for f in provided if any(tok in f.lower() for tok in FORBIDDEN_SUBSTRINGS)]
    if forbidden_hits:
        raise HTTPException(
            status_code=422,
            detail=f"Features interdites (leakage potentiel) : {sorted(forbidden_hits)}",
        )

    missing = set(container.cols) - provided
    if missing:
        raise HTTPException(status_code=422, detail=f"Features manquantes : {sorted(missing)}")

    try:
        feat = {k: float(v) for k, v in payload.features.items() if k in container.cols}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Feature invalide : {exc}") from exc

    request_id = request.headers.get("X-Request-ID", None)
    result = predictor.predict(
        feat,
        customer_id=payload.customer_id,
        n_past_loans=payload.n_past_loans,
        request_id=request_id,
        actor=client,
    )

    monitor = getattr(request.app.state, "drift_monitor", None)
    if monitor is not None:
        monitor.observe(feat)

    duration = time.perf_counter() - start
    record_score(
        model_version=request.app.state.model_version,
        probability=result.probability,
        duration_seconds=duration,
    )
    logger.info(
        "api.predict.done",
        customer_id=payload.customer_id,
        decision=result.decision,
        probability=round(result.probability, 6),
        confidence=result.confidence,
        client=client,
    )

    return ScoreResponse(**result.to_dict())


@router_v1.post(
    "/lending-club/score",
    response_model=LendingClubScoreResponse,
    tags=["scoring-validation"],
    summary="Démonstration du champion validé sur données publiques réelles (pas le pilote CIF)",
)
async def lending_club_score(
    payload: LendingClubScoreRequest,
    request: Request,
    client: str = Depends(_client_identity),
) -> LendingClubScoreResponse:
    """Scoring via le modèle ``lending_club_champion`` (voir docs/validation/lending-club-benchmark.md).

    Endpoint de démonstration méthodologique, distinct de ``/v1/predict`` (pilote CIF synthétique) :
    schéma de features et policy différents, jamais à confondre dans une même réponse.
    """
    await _enforce_rate_limit(request, client)
    lc_predictor: LendingClubPredictor | None = getattr(request.app.state, "lending_club_predictor", None)
    if lc_predictor is None:
        raise HTTPException(status_code=503, detail="Modèle lending_club_champion non chargé")

    provided = set(payload.features)
    missing = set(LC_FEATURES) - provided
    if missing:
        raise HTTPException(status_code=422, detail=f"Features manquantes : {sorted(missing)}")

    request_id = request.headers.get("X-Request-ID", None)
    result = lc_predictor.predict(payload.features, request_id=request_id, actor=client)
    logger.info(
        "api.lending_club_score.done",
        decision=result.decision,
        probability=round(result.probability, 6),
        client=client,
    )
    return LendingClubScoreResponse(**result.to_dict())
