"""Intégration modèle + décision — service de prédiction réutilisable (Semaine 3).

`Predictor` orchestre : modèle (probabilité) → confiance → décision (DecisionEngine) →
journalisation d'audit atomique. Il est injecté dans l'API (``app.state.predictor``),
couche métier unique pour le scoring par l'API et les pipelines (reproductibilité totale).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pandas as pd

from config.schema import FeatureConfig
from services.audit_service import AuditService
from services.confidence import prediction_confidence
from services.decision_engine import DecisionEngine

DecisionThresholds = tuple[float, float, float]

THRESHOLD_DEFAULTS: DecisionThresholds = (0.10, 0.25, 0.50)


def risk_class(probability: float) -> str:
    """Classe de risque dérivée de la probabilité (lecture humaine, tracée en BDD)."""
    if probability < 0.1:
        return "faible"
    if probability < 0.25:
        return "moyen"
    if probability < 0.5:
        return "élevé"
    return "critique"


@dataclass
class PredictionResult:
    """Résultat complet d'une prédiction — entièrement traçable (miroir de la BDD)."""

    customer_id: int
    probability: float
    score: int
    decision: str
    policy_hit: str
    confidence: float
    risk_class: str
    model_version: str
    request_id: str
    features: dict[str, float] = field(default_factory=dict)
    factors: dict[str, float] = field(default_factory=dict)
    prediction_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "probability": round(self.probability, 6),
            "score": self.score,
            "risk_class": self.risk_class,
            "decision": self.decision,
            "confidence": round(self.confidence, 6),
            "factors": {k: round(v, 6) for k, v in self.factors.items()},
            "policy_hit": self.policy_hit,
        }


class Predictor:
    """Service métier unifié : modèle → confiance → décision → audit (atomique)."""

    def __init__(
        self,
        engine: DecisionEngine,
        model_version: str,
        *,
        audit: AuditService | None = None,
        thresholds: DecisionThresholds = THRESHOLD_DEFAULTS,
    ) -> None:
        self.engine = engine
        self.model_version = model_version
        self.audit = audit
        self.thresholds = thresholds

    def predict(
        self,
        features: dict[str, float],
        customer_id: int,
        n_past_loans: int,
        *,
        request_id: str | None = None,
        actor: str | None = None,
    ) -> PredictionResult:
        """Calcule PD, confiance et décision pour un client et journalise (si audit actif)."""
        if self.engine.model is None:
            raise RuntimeError("Predictor requiert un modèle chargé (engine.model).")

        rid = request_id or str(uuid4())
        # XGBoost exige que les colonnes soient dans l'ordre d'entrainement ;
        # on re-ordonne explicitement (independant de l'ordre du client) pour
        # eviter un mismatch de feature_names (500 sinon).
        cols = [c for fam in FeatureConfig().families.values() for c in fam]
        feat_df = pd.DataFrame([features]).astype(float)[cols]
        probability = float(self.engine.model.predict_proba(feat_df)[0, 1])
        confidence = prediction_confidence(probability, self.thresholds)
        outcome = self.engine.decide(
            probability=probability,
            n_past_loans=n_past_loans,
            customer_id=customer_id,
            confidence=confidence,
        )

        prediction_id = None
        if self.audit is not None:
            prediction_id = self.audit.record_prediction(
                actor=actor,
                customer_id=customer_id,
                request_id=rid,
                model_version=self.model_version,
                probability=probability,
                score=outcome.score,
                decision=outcome.decision.value,
                confidence=confidence,
                policy_hit=outcome.policy_hit,
                risk_class=risk_class(probability),
                features=features,
                factors=outcome.factors,
            )

        return PredictionResult(
            customer_id=customer_id,
            probability=probability,
            score=outcome.score,
            decision=outcome.decision.value,
            policy_hit=outcome.policy_hit,
            confidence=confidence,
            risk_class=risk_class(probability),
            model_version=self.model_version,
            request_id=rid,
            features=features,
            factors=outcome.factors,
            prediction_id=prediction_id,
        )
