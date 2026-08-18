"""Contrat de la feature `max_dpd` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="max_dpd",
    version="1.0.0",
    family="history",
    description="Retard maximal de paiement en jours (jours passés dus).",
    bounds=(0.0, 90.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
