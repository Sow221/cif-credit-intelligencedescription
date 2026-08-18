"""Logging structuré (structlog) aligné sur les pratiques d'observabilité."""

from __future__ import annotations

import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog pour une sortie console structurée et lisible en dev."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(min_level=level.upper()),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Retourne un logger structuré typé."""
    configure_logging()
    return structlog.get_logger(name)
