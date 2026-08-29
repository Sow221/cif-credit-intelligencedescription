"""Contrat de la feature `loan_to_income_ratio` (famille context).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="loan_to_income_ratio",
    version="1.0.0",
    family="context",
    description="Ratio demande de crédit courante / revenu mensuel.",
    bounds=(0.0, None),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
