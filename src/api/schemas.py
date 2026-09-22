"""Schémas d'entrée/sortie de l'API de scoring (Pydantic v2).

Validation stricte : ``extra="forbid"`` sur tous les modèles (exigence cabinet) —
toute donnée inconnue est rejetée avec un 422 détaillé.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DecisionType = Literal["APPROBATION", "AJUSTEMENT", "REVUE_HUMAINE", "REFUS"]


class StrictModel(BaseModel):
    """Base commune : rejette tout champ inconnu."""

    model_config = ConfigDict(extra="forbid")


class ScoreRequest(StrictModel):
    """Requête de scoring — une demande de crédit d'un client."""

    customer_id: int = Field(..., description="Identifiant unique du client")
    features: dict[str, float] = Field(..., description="Features disponibles avant décision")
    n_past_loans: int = Field(default=0, ge=0, description="Nombre de prêts passés (historique)")


class ScoreResponse(StrictModel):
    """Réponse du scoring, alignée sur M07 (score, probabilité, classe, explications, décision)."""

    model_version: str = Field(..., description="Version du modèle servie")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilité estimée de défaut")
    score: int = Field(..., ge=0, le=100, description="Score de risque (100 = risque minimal)")
    risk_class: str = Field(..., description="Classe de risque (faible/moyen/élevé/critique)")
    decision: DecisionType = Field(..., description="Décision proposée (human-in-the-loop)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Niveau de confiance")
    factors: dict[str, float] = Field(default_factory=dict, description="Facteurs explicatifs")
    policy_hit: str = Field(..., description="Règle de policy appliquée")


class LendingClubScoreRequest(StrictModel):
    """Requête de scoring — démonstration du champion validé sur données publiques réelles.

    Distinct de ``ScoreRequest`` (pilote CIF synthétique) : schéma et policy propres au modèle
    ``lending_club_champion`` (voir ``docs/validation/lending-club-benchmark.md``).
    """

    features: dict[str, float | str | None] = Field(
        ...,
        description=(
            "18 features Lending Club (voir features.lending_club.FEATURES). "
            "``null`` est accepté pour une variable numérique inconnue (le modèle impute "
            "explicitement les valeurs manquantes) — jamais ``NaN`` littéral, invalide en JSON strict."
        ),
    )


class LendingClubScoreResponse(StrictModel):
    """Réponse du scoring Lending Club — PD, décision par coût, explications par variable."""

    model_version: str = Field(..., description="URI/alias MLflow du modèle servi")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilité estimée de défaut")
    decision: Literal["APPROBATION", "REVUE_HUMAINE", "REFUS"] = Field(
        ..., description="Décision autour du seuil de coût (marge = revue humaine)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Distance normalisée au seuil")
    threshold: float = Field(..., description="Seuil de PD utilisé (optimal par coût, voir ADR)")
    factors: dict[str, float] = Field(
        default_factory=dict, description="Contribution de chaque variable au score (log-odds, exact)"
    )


class HealthResponse(StrictModel):
    """État de santé du service."""

    status: str
    model_version: str | None = None
    uptime_seconds: float = 0.0


class TokenRequest(StrictModel):
    """Credentials d'un agent/partenaire demandant un jeton."""

    client_id: str = Field(..., min_length=1, description="Identifiant du client")
    client_secret: str = Field(..., min_length=1, description="Secret du client")


class TokenResponse(StrictModel):
    """Réponse d'authentification — jeton JWT à envoyer dans Authorization: Bearer."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(..., ge=0, description="Durée de validité en secondes")
