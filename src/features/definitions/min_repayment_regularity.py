"""Contrat de la feature `min_repayment_regularity` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="min_repayment_regularity",
    version="1.0.0",
    family="history",
    description="Régularité de remboursement minimale observée sur les prêts (0..1).",
    bounds=(0.0, 1.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
