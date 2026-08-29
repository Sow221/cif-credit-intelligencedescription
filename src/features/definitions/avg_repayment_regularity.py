"""Contrat de la feature `avg_repayment_regularity` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="avg_repayment_regularity",
    version="1.0.0",
    family="history",
    description="Régularité moyenne de remboursement (0 = irrégulier, 1 = très régulier).",
    bounds=(0.0, 1.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
