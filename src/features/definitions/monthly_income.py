"""Contrat de la feature `monthly_income` (famille profile_income).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="monthly_income",
    version="1.0.0",
    family="profile_income",
    description="Revenu mensuel déclaré (FCFA).",
    bounds=(20000.0, 2000000.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
