"""Contrat de la feature `current_savings` (famille savings).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="current_savings",
    version="1.0.0",
    family="savings",
    description="Solde d'épargne courant (FCFA).",
    bounds=(0.0, 8000000.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
