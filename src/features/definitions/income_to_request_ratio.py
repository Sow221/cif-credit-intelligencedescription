"""Contrat de la feature `income_to_request_ratio` (famille context).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="income_to_request_ratio",
    version="1.0.0",
    family="context",
    description="Ratio revenu mensuel / montant demandé (clippé).",
    bounds=(0.0, 10.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
