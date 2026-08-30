"""Métriques Prometheus exposées par l'API de scoring (qualité du service, drift)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Métriques HTTP (qualité de service) — standard Prometheus
REQUEST_COUNT = Counter(
    "cif_http_requests_total",
    "Nombre de requêtes HTTP traitées",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "cif_http_request_duration_seconds",
    "Latence des requêtes HTTP",
    ["endpoint"],
)

# Compteur de requêtes de scoring par décision
SCORE_REQUESTS = Counter(
    "cif_score_requests_total",
    "Nombre de requêtes de scoring reçues",
    ["model_version"],
)

# Distribution des probabilités prédites
SCORE_PROBABILITY = Histogram(
    "cif_score_probability",
    "Distribution des probabilités de défaut prédites",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Durée des requêtes de scoring
SCORE_DURATION = Histogram(
    "cif_score_duration_seconds",
    "Temps de traitement d'une requête de scoring",
)

# Dérive observée (taux de features en drift)
DRIFT_RATIO = Gauge(
    "cif_drift_ratio",
    "Taux de features en dérive entre référence et production",
)

# Alertes de drift déclenchées
DRIFT_ALERTS = Counter(
    "cif_drift_alerts_total",
    "Nombre d'alertes de dérive déclenchées",
)


def record_score(model_version: str, probability: float, duration_seconds: float) -> None:
    """Enregistre une prédiction dans les métriques Prometheus."""
    SCORE_REQUESTS.labels(model_version=model_version).inc()
    SCORE_PROBABILITY.observe(probability)
    SCORE_DURATION.observe(duration_seconds)


def record_request(method: str, endpoint: str, status: int, duration_seconds: float) -> None:
    """Enregistre une requête HTTP dans les métriques Prometheus."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration_seconds)
