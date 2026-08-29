"""Contrat de la feature `historical_default_rate` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="historical_default_rate",
    version="1.0.0",
    family="history",
    description="Taux de défaut historique du client (n_defaults / n_loans).",
    bounds=(0.0, 1.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
