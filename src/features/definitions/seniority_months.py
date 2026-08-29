"""Contrat de la feature `seniority_months` (famille profile_income).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="seniority_months",
    version="1.0.0",
    family="profile_income",
    description="Ancienneté du client dans l'institution (mois).",
    bounds=(0.0, 240.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
