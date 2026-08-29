"""Contrat de la feature `current_loan_duration` (famille profile_income).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="current_loan_duration",
    version="1.0.0",
    family="profile_income",
    description="Durée de la demande courante (mois).",
    bounds=(0.0, 24.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
