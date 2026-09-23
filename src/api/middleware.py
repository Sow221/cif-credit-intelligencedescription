"""Middleware FastAPI — Request ID et Rate limiting par client (exigence cabinet, Semaine 2).

- ``RequestIDMiddleware`` : garantit un identifiant de requête bout-en-bout (header
  ``X-Request-ID``, contextvars structlog) pour la traçabilité de l'audit trail.
- ``RateLimiter`` / ``RedisRateLimiter`` : fenêtre glissante de 60 s par client ; renvoie 429
  au-delà. Deux implémentations d'un même protocole (``RateLimiterProtocol``) : la première
  compte en mémoire du processus — correcte pour une seule instance, incorrecte dès qu'il y a
  plusieurs réplicas (chacun compte séparément, le quota réel devient N fois plus permissif que
  configuré). ``RedisRateLimiter`` compte dans un magasin partagé, correcte à toute échelle.
  ``create_app`` choisit l'une ou l'autre selon qu'un ``CIF_REDIS_URL`` est configuré — voir
  ``api/app.py``.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from redis.commands.core import AsyncScript

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from monitoring.metrics import record_request


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attribue et propage un identifiant de requête unique."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["X-Request-ID"] = request_id
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Instrumente chaque requête HTTP dans Prometheus (compteur + latence)."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        record_request(request.method, request.url.path, response.status_code, duration)
        return response


class RateLimiterProtocol(Protocol):
    """Contrat commun aux deux implémentations — permet à l'API de les utiliser indifféremment."""

    rate_per_minute: int

    async def allow(self, client: str) -> bool: ...


class RateLimiter:
    """Rate limiting par fenêtre glissante, en mémoire du processus (thread-safe).

    Correct pour une seule instance. Avec plusieurs réplicas (K8s, plusieurs workers), chaque
    processus compte séparément : le quota réel effectif devient ``rate_per_minute`` fois le
    nombre d'instances, pas celui configuré. Utiliser ``RedisRateLimiter`` au-delà d'une instance.
    """

    def __init__(self, rate_per_minute: int = 120) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute doit être strictement positif")
        self.rate_per_minute = rate_per_minute
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    async def allow(self, client: str) -> bool:
        """True si la requête est autorisée, False si le quota du client est dépassé."""
        now = time.monotonic()
        window_start = now - 60.0
        with self._lock:
            hits = [t for t in self._hits.get(client, []) if t > window_start]
            if len(hits) >= self.rate_per_minute:
                self._hits[client] = hits
                return False
            hits.append(now)
            self._hits[client] = hits
            return True


# Script Lua : lit + purge la fenêtre + décide + enregistre en une seule opération atomique côté
# serveur Redis. Sans ça, deux requêtes concurrentes sur deux instances différentes pourraient
# toutes les deux lire "quota non atteint" avant que l'une des deux n'écrive son propre hit
# (race condition classique du "check-then-act" à travers le réseau).
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
  redis.call('EXPIRE', key, window)
  return 0
end
redis.call('ZADD', key, now, now .. '-' .. redis.call('INCR', key .. ':seq'))
redis.call('EXPIRE', key, window)
redis.call('EXPIRE', key .. ':seq', window)
return 1
"""


class RedisRateLimiter:
    """Rate limiting par fenêtre glissante dans Redis — correct à travers plusieurs instances.

    Même fenêtre de 60 s et même sémantique que ``RateLimiter`` ; un seul magasin partagé,
    donc un client qui frappe l'instance A puis l'instance B est compté une seule fois, pas deux.
    """

    def __init__(self, redis_client: Redis, rate_per_minute: int = 120, window_seconds: int = 60) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute doit être strictement positif")
        self.rate_per_minute = rate_per_minute
        self.window_seconds = window_seconds
        self._redis = redis_client
        self._script: AsyncScript | None = None

    async def allow(self, client: str) -> bool:
        """True si la requête est autorisée, False si le quota du client est dépassé."""
        if self._script is None:
            self._script = self._redis.register_script(_SLIDING_WINDOW_LUA)
        now = time.time()
        result = await self._script(
            keys=[f"cif:ratelimit:{client}"],
            args=[now, self.window_seconds, self.rate_per_minute],
        )
        return bool(result)
