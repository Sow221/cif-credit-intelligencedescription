"""Sécurité de l'API — authentification JWT et credentials clients (exigence cabinet : JWT).

Implémentation HS256 via PyJWT. Les secrets ne sont jamais commités : ils viennent de
l'environnement (``CIF_API_CLIENT_ID`` / ``CIF_API_CLIENT_SECRET`` / ``CIF__JWT_SECRET``)
ou du ``.env`` (voir ``.env.example``).
"""

from __future__ import annotations

import hmac
import os
import time

import jwt
from pydantic import BaseModel, Field

DEFAULT_CLIENT_ID = "cif-agent"
DEFAULT_CLIENT_SECRET = "change-me-in-production"


class TokenPayload(BaseModel):
    """Revendications du jeton JWT (minimales et vérifiables)."""

    sub: str = Field(..., description="Identifiant du client autorisé")
    iat: float = Field(..., description="Timestamp d'émission (epoch)")
    exp: float = Field(..., description="Timestamp d'expiration (epoch)")


def _client_id() -> str:
    return os.environ.get("CIF_API_CLIENT_ID", DEFAULT_CLIENT_ID)


def _client_secret() -> str:
    return os.environ.get("CIF_API_CLIENT_SECRET", DEFAULT_CLIENT_SECRET)


def assert_production_secrets(jwt_secret: str | None) -> None:
    """Refuse de démarrer en production avec un secret par défaut ou non configuré.

    Actif quand ``CIF_ENV=production``. En dev/test, un secret JWT aléatoire est généré à
    chaque démarrage (jamais une valeur connue) et le secret client par défaut reste toléré.
    """
    if os.environ.get("CIF_ENV", "dev").lower() != "production":
        return
    problems: list[str] = []
    if not jwt_secret or jwt_secret == DEFAULT_CLIENT_SECRET:
        problems.append("CIF_JWT_SECRET absent ou égal à la valeur par défaut")
    if _client_secret() == DEFAULT_CLIENT_SECRET:
        problems.append("CIF_API_CLIENT_SECRET vaut la valeur par défaut")
    if problems:
        raise RuntimeError("Configuration de production invalide : " + " ; ".join(problems))


def check_credentials(client_id: str, client_secret: str) -> bool:
    """Vérifie les credentials d'un client (comparaison à temps constant)."""
    return hmac.compare_digest(client_id, _client_id()) and hmac.compare_digest(client_secret, _client_secret())


def create_access_token(
    client_id: str,
    secret: str,
    ttl_minutes: int = 60,
    algorithm: str = "HS256",
) -> str:
    """Émet un jeton JWT signé HS256 pour un client authentifié."""
    now = time.time()
    payload = TokenPayload(sub=client_id, iat=now, exp=now + ttl_minutes * 60).model_dump()
    return jwt.encode(payload, secret, algorithm=algorithm)


def verify_token(token: str, secret: str, algorithm: str = "HS256") -> TokenPayload:
    """Vérifie la signature et l'expiration ; lève jwt.InvalidTokenError si invalide."""
    data: dict[str, object] = jwt.decode(token, secret, algorithms=[algorithm])
    return TokenPayload(sub=str(data["sub"]), iat=float(str(data["iat"])), exp=float(str(data["exp"])))
