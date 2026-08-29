"""Contrat de la feature `savings_std_24m` (famille savings).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="savings_std_24m",
    version="1.0.0",
    family="savings",
    description="Écart-type des soldes d'épargne sur les 24 derniers mois (FCFA).",
    bounds=(0.0, 3000000.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
