"""Contrat de la feature `age` (famille profile_income).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="age",
    version="1.0.0",
    family="profile_income",
    description="Age du client en années.",
    bounds=(18.0, 75.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
