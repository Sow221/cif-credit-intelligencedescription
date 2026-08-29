"""Contrat de la feature `savings_to_income_ratio` (famille savings).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="savings_to_income_ratio",
    version="1.0.0",
    family="savings",
    description="Ratio épargne courante / revenu mensuel.",
    bounds=(0.0, None),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
