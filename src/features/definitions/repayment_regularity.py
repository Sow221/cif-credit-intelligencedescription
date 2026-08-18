"""Contrat de la feature `repayment_regularity` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="repayment_regularity",
    version="1.0.0",
    family="history",
    description="Régularité moyenne de remboursement (0..1).",
    bounds=(0.0, 1.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
