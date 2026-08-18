"""Contrat de la feature `location_num` (famille profile_income).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="location_num",
    version="1.0.0",
    family="profile_income",
    description="Zone géographique encodée (urbain/periurbain/rural).",
    bounds=(-1.0, 2.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
