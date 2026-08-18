"""Contrat de la feature `savings_volatility` (famille savings).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="savings_volatility",
    version="1.0.0",
    family="savings",
    description="Volatilité normalisée de l'épargne (0..1).",
    bounds=(0.01, 1.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
