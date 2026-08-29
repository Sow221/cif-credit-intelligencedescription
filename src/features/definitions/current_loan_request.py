"""Contrat de la feature `current_loan_request` (famille profile_income).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="current_loan_request",
    version="1.0.0",
    family="profile_income",
    description="Montant de la demande de crédit courante (FCFA).",
    bounds=(20000.0, 20000000.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
