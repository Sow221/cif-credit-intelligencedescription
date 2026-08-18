"""Contrat de la feature `gender_num` (famille profile_income).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="gender_num",
    version="1.0.0",
    family="profile_income",
    description="Sexe encodé numériquement (0=M, 1=F).",
    bounds=(0.0, 1.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
