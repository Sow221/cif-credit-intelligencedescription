"""Decision Engine — sépare strictement Model / Policy / Workflow / Decision (cahier des charges §64).

Le modèle produit une probabilité ; la *policy* applique des seuils et règles (dont thin-file) ;
le *workflow* détermine qui doit valider ; la *décision* finale est traçable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd


class Decision(StrEnum):
    """Décisions possibles du moteur."""

    APPROBATION = "APPROBATION"
    AJUSTEMENT = "AJUSTEMENT"
    REVUE_HUMAINE = "REVUE_HUMAINE"
    REFUS = "REFUS"


@dataclass
class DecisionOutcome:
    """Résultat d'une décision — entièrement traçable."""

    customer_id: int
    probability: float
    score: int
    decision: Decision
    policy_hit: str
    factors: dict[str, float]
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "customer_id": self.customer_id,
            "probability": round(self.probability, 6),
            "score": self.score,
            "decision": self.decision.value,
            "policy_hit": self.policy_hit,
            "factors": self.factors,
            "confidence": round(self.confidence, 6),
        }


@dataclass
class DecisionPolicy:
    """Policy de décision (seuils et règles), configurable — indépendante du modèle."""

    approve_threshold: float = 0.10
    review_threshold: float = 0.25
    hard_reject_threshold: float = 0.50
    thin_file_review: bool = True
    thin_file_max_loans: int = 1

    def apply(self, probability: float, n_past_loans: int) -> tuple[Decision, str]:
        """Applique les règles métier (Policy) sur la probabilité seule."""
        if probability < self.approve_threshold:
            return Decision.APPROBATION, "prob < approve"
        if probability < self.review_threshold:
            return Decision.AJUSTEMENT, "approve <= prob < review"
        if probability < self.hard_reject_threshold:
            return Decision.REVUE_HUMAINE, "review <= prob < hard_reject"
        return Decision.REFUS, "prob >= hard_reject"


def _score_from_probability(probability: float) -> int:
    """Transforme une probabilité en score 0-100 (100 = risque minimal)."""
    return round((1.0 - probability) * 100.0)


class DecisionEngine:
    """Moteur complet : modèle ≠ policy ≠ workflow ≠ décision."""

    def __init__(self, policy: DecisionPolicy | None = None, model: Any | None = None) -> None:
        self.policy = policy or DecisionPolicy()
        self.model = model

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Retourne la probabilité de défaut pour un batch de features."""
        if self.model is None:
            raise RuntimeError("DecisionEngine requiert un modèle entraîné.")
        probs = self.model.predict_proba(features.astype(float))[:, 1]
        return pd.DataFrame({"probability": probs}, index=features.index)

    def decide(
        self,
        probability: float,
        n_past_loans: int,
        customer_id: int,
        factors: dict[str, float] | None = None,
        confidence: float = 1.0,
    ) -> DecisionOutcome:
        """Décide pour un client donné (human-in-the-loop quand nécessaire)."""
        decision, hit = self.policy.apply(probability, n_past_loans)

        # Règle métier : les thin-file doivent être revus par un humain (audit Phase E/F)
        if (
            self.policy.thin_file_review
            and n_past_loans <= self.policy.thin_file_max_loans
            and decision in (Decision.APPROBATION, Decision.AJUSTEMENT)
        ):
            decision = Decision.REVUE_HUMAINE
            hit = f"{hit} + thin_file_review"

        return DecisionOutcome(
            customer_id=customer_id,
            probability=float(probability),
            score=_score_from_probability(float(probability)),
            decision=decision,
            policy_hit=hit,
            factors=factors or {},
            confidence=float(confidence),
        )

    def decide_batch(
        self,
        probs: pd.DataFrame,
        n_past_loans: pd.Series,
        customer_ids: pd.Series,
    ) -> list[DecisionOutcome]:
        """Applique la décision sur un batch, avec la liste des n_past_loans."""
        outcomes: list[DecisionOutcome] = []
        for (_, row), loans, cid in zip(probs.iterrows(), n_past_loans, customer_ids, strict=True):
            outcomes.append(self.decide(float(row["probability"]), int(loans), int(cid)))
        return outcomes
