"""Contrat de la feature `active_loans` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="active_loans",
    version="1.0.0",
    family="history",
    description="Nombre de prêts en cours (ouverts).",
    bounds=(0.0, 8.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
