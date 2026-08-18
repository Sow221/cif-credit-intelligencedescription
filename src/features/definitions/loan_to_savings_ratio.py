"""Contrat de la feature `loan_to_savings_ratio` (famille savings).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="loan_to_savings_ratio",
    version="1.0.0",
    family="savings",
    description="Ratio demande de crédit / épargne courante.",
    bounds=(0.0, 200.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
