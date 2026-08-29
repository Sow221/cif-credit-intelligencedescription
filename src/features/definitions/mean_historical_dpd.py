"""Contrat de la feature `mean_historical_dpd` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="mean_historical_dpd",
    version="1.0.0",
    family="history",
    description="DPD moyen (jours de retard) sur l'historique de prêts.",
    bounds=(0.0, 365.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
