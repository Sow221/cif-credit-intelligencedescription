"""Contrat de la feature `savings_trend` (famille savings).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="savings_trend",
    version="1.0.0",
    family="savings",
    description="Pente mensuelle de l'épargne (régression linéaire).",
    bounds=(-100000.0, 100000.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
