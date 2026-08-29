"""Contrat de la feature `seniority_years` (famille context).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="seniority_years",
    version="1.0.0",
    family="context",
    description="Ancienneté du client dans l'institution, en années.",
    bounds=(0.0, 20.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
