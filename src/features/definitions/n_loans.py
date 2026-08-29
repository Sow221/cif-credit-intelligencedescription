"""Contrat de la feature `n_loans` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="n_loans",
    version="1.0.0",
    family="history",
    description="Nombre total de prêts historiques du client (agrégé sur loans).",
    bounds=(0.0, 100.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
