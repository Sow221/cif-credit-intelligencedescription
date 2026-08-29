"""Contrat de la feature `max_historical_dpd` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="max_historical_dpd",
    version="1.0.0",
    family="history",
    description="DPD maximal (jours de retard) observé sur l'historique de prêts.",
    bounds=(0.0, 365.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
