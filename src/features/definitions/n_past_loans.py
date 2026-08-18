"""Contrat de la feature `n_past_loans` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="n_past_loans",
    version="1.0.0",
    family="history",
    description="Nombre de prêts passés remboursés/en cours (comptage réel de l'historique).",
    bounds=(0.0, 8.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
