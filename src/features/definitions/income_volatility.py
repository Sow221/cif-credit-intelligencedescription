"""Contrat de la feature `income_volatility` (famille profile_income).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="income_volatility",
    version="1.0.0",
    family="profile_income",
    description="Volatilité du revenu (0=stable, 1=très volatile).",
    bounds=(0.05, 1.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
