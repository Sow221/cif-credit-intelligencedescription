"""Service de scoring Lending Club — sert le champion enregistré (démonstration de la méthode).

Distinct du ``Predictor`` du pilote CIF (modèle synthétique, 25 features, policy CIF §64) :
ce service sert le modèle validé sur données publiques réelles (`ADR 0001-0003`), avec son
propre schéma de features (18) et sa propre policy (seuil unique par coût, pas les 4 seuils
CIF). Les deux ne doivent jamais être confondus dans les réponses de l'API.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd

from features.lending_club import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from models.scoring import ScoringModel
from services.audit_service import AuditEvent, AuditService

DEFAULT_REVIEW_MARGIN = 0.03


@dataclass
class LendingClubResult:
    """Résultat d'un scoring — miroir de ``PredictionResult`` mais schéma Lending Club."""

    probability: float
    decision: str
    confidence: float
    threshold: float
    model_version: str
    request_id: str
    factors: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "probability": round(self.probability, 6),
            "decision": self.decision,
            "confidence": round(self.confidence, 6),
            "threshold": round(self.threshold, 4),
            "factors": {k: round(v, 6) for k, v in self.factors.items()},
        }


class LendingClubPredictor:
    """Charge le champion depuis le registre MLflow et sert des prédictions expliquées."""

    def __init__(
        self,
        model: ScoringModel,
        model_version: str,
        threshold: float,
        *,
        review_margin: float = DEFAULT_REVIEW_MARGIN,
        audit: AuditService | None = None,
    ) -> None:
        self.model = model
        self.model_version = model_version
        self.threshold = threshold
        self.review_margin = review_margin
        self.audit = audit

    @classmethod
    def from_registry(
        cls,
        model_uri: str = "models:/lending_club_champion@champion",
        *,
        default_threshold: float = 0.15,
        audit: AuditService | None = None,
    ) -> LendingClubPredictor:
        """Charge le pyfunc (registre MLflow *ou* dossier local exporté) et son seuil de coût.

        Deux cas, selon l'URI :
        - ``models:/nom@alias`` (stack locale avec serveur MLflow) : le seuil est lu en tag sur
          la version, via le client du registre.
        - un chemin local (artefact « cuit » dans une image Docker par
          ``cli-export-lc-model``, aucun serveur MLflow requis) : le seuil est lu dans
          ``cost_threshold.txt``, écrit à côté par cette même commande d'export.
        """
        loaded = mlflow.pyfunc.load_model(model_uri)
        scoring_model = loaded.unwrap_python_model().model

        threshold = default_threshold
        if model_uri.startswith("models:/"):
            try:
                client = mlflow.tracking.MlflowClient()
                name, alias = model_uri.removeprefix("models:/").split("@")
                mv = client.get_model_version_by_alias(name, alias)
                threshold = float(mv.tags.get("cost_threshold", default_threshold))
            except Exception:  # pragma: no cover — dégrade sur le seuil par défaut documenté
                pass
        else:
            threshold_file = Path(model_uri) / "cost_threshold.txt"
            if threshold_file.is_file():
                threshold = float(threshold_file.read_text(encoding="utf-8").strip())
        return cls(scoring_model, model_uri, threshold, audit=audit)

    def decide(self, probability: float) -> str:
        """Décision à trois niveaux autour du seuil de coût (voir ``docs/adr``)."""
        lo, hi = self.threshold - self.review_margin, self.threshold + self.review_margin
        if probability < lo:
            return "APPROBATION"
        if probability > hi:
            return "REFUS"
        return "REVUE_HUMAINE"

    def predict(
        self, features: dict[str, float | str], *, request_id: str | None = None, actor: str | None = None
    ) -> LendingClubResult:
        """Calcule PD, explication et décision pour une demande ; journalise si l'audit est actif."""
        rid = request_id or str(uuid4())
        row: dict[str, list[float | str]] = {}
        for c in NUMERIC_FEATURES:
            row[c] = [float(features[c])] if features.get(c) is not None else [np.nan]
        for c in CATEGORICAL_FEATURES:
            row[c] = [str(features.get(c, "unknown"))]
        df = pd.DataFrame(row)

        probability = float(self.model.predict_proba(df)[0])
        decision = self.decide(probability)
        confidence = float(np.clip(abs(probability - self.threshold) / max(self.review_margin, 1e-6), 0.0, 1.0))

        factors: dict[str, float] = {}
        with contextlib.suppress(NotImplementedError):
            factors = self.model.explain(df).drop(columns=["bias"], errors="ignore").iloc[0].to_dict()

        if self.audit is not None:
            self.audit.record(
                AuditEvent.PREDICTION_REQUESTED,
                actor=actor,
                request_id=rid,
                model_version=self.model_version,
                status=decision,
            )

        return LendingClubResult(
            probability=probability,
            decision=decision,
            confidence=confidence,
            threshold=self.threshold,
            model_version=self.model_version,
            request_id=rid,
            factors=factors,
        )
