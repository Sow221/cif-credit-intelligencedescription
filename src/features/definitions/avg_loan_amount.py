"""Contrat de la feature `avg_loan_amount` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="avg_loan_amount",
    version="1.0.0",
    family="history",
    description="Montant moyen des prêts historiques (FCFA).",
    bounds=(0.0, 15000000.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
