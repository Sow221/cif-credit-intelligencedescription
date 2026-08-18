"""Contrat de la feature `payments_on_time_ratio` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="payments_on_time_ratio",
    version="1.0.0",
    family="history",
    description="Proportion de paiements effectués à temps.",
    bounds=(0.0, 1.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
