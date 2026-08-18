"""Contrat de la feature `avg_savings_24m` (famille savings).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="avg_savings_24m",
    version="1.0.0",
    family="savings",
    description="Solde d'épargne moyen sur 24 mois (FCFA).",
    bounds=(0.0, 7000000.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
