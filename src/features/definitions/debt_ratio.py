"""Contrat de la feature `debt_ratio` (famille context).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="debt_ratio",
    version="1.0.0",
    family="context",
    description="Ratio demande / revenu annuel (endettement, clippé).",
    bounds=(0.0, 20.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
