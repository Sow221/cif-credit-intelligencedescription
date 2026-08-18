"""Contrat de la feature `sector_num` (famille profile_income).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="sector_num",
    version="1.0.0",
    family="profile_income",
    description="Secteur d'activité encodé (agriculture/commerce/services/elevage).",
    bounds=(-1.0, 3.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
