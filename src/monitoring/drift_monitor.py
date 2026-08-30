"""Monitoring de drift temps réel — Evidently sur le trafic de production.

Le ``DriftMonitor`` bufferise les features des requêtes de scoring et recalcule
périodiquement le taux de drift (DataDriftPreset) vs une baseline de référence,
alimentant les jauges Prometheus ``cif_drift_ratio`` / ``cif_drift_alerts_total``.
La baseline est la distribution d'entraînement (générée ici à partir du même
générateur synthétique que l'entraînement — données honnêtement labellisées).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import pandas as pd

from config.schema import DataConfig, FeatureConfig
from data.synthetic import generate_datasets
from features.builder import build_features
from monitoring.drift import check_drift_alert, compute_drift_ratio
from monitoring.metrics import DRIFT_ALERTS, DRIFT_RATIO
from utils.logging import get_logger

logger = get_logger(__name__)

REFERENCE_PATH = Path("data/monitoring/reference.parquet")
DRIFT_COMPUTE_INTERVAL_SECONDS = 60
DRIFT_MIN_SAMPLES = 50


def build_reference_baseline(feature_cols: list[str], path: Path = REFERENCE_PATH) -> pd.DataFrame:
    """Construit et persiste la baseline de référence (distribution d'entraînement)."""
    ds = generate_datasets(DataConfig())
    feat = build_features(ds.customers, ds.loans, ds.savings, FeatureConfig())
    reference = feat[feature_cols].astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(path, index=False)
    logger.info("monitoring.baseline.built", rows=len(reference), path=str(path))
    return reference


def load_reference(feature_cols: list[str], path: Path = REFERENCE_PATH) -> pd.DataFrame:
    """Charge la baseline ; la (re)génère si absente."""
    if path.exists():
        return pd.read_parquet(path)[feature_cols].astype(float)
    return build_reference_baseline(feature_cols, path)


class DriftMonitor:
    """Bufferise les features de production et expose le drift via Prometheus."""

    def __init__(
        self,
        reference: pd.DataFrame,
        feature_cols: list[str],
        max_buffer: int = 500,
        min_samples: int = DRIFT_MIN_SAMPLES,
    ) -> None:
        self.reference = reference[feature_cols].astype(float)
        self.cols = feature_cols
        self.buffer: deque[dict[str, float]] = deque(maxlen=max_buffer)
        self.min_samples = min_samples

    def observe(self, features: dict[str, float]) -> None:
        """Enregistre une observation de production (lignes complètes uniquement)."""
        row: dict[str, float] = {}
        for c in self.cols:
            v = features.get(c)
            if not isinstance(v, (int, float)):
                return
            row[c] = float(v)
        self.buffer.append(row)

    def compute(self) -> float | None:
        """Recalcule le drift et met à jour les jauges Prometheus. Retourne le ratio."""
        if len(self.buffer) < self.min_samples:
            return None
        current = pd.DataFrame(self.buffer)[self.cols].astype(float)
        ratio = compute_drift_ratio(self.reference, current)
        DRIFT_RATIO.set(ratio)
        if check_drift_alert({"drift_ratio": ratio}):
            DRIFT_ALERTS.inc()
        logger.info("monitoring.drift.computed", drift_ratio=ratio, samples=len(current))
        return ratio
