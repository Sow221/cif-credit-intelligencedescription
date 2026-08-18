"""Middleware FastAPI — Request ID et Rate limiting par client (exigence cabinet, Semaine 2).

- ``RequestIDMiddleware`` : garantit un identifiant de requête bout-en-bout (header
  ``X-Request-ID``, contextvars structlog) pour la traçabilité de l'audit trail.
- ``RateLimiter`` : fenêtre glissante de 60 s par client ; renvoie 429 au-delà.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


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


class RateLimiter:
    """Rate limiting par fenêtre glissante (thread-safe)."""

    def __init__(self, rate_per_minute: int = 120) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute doit être strictement positif")
        self.rate_per_minute = rate_per_minute
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
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
