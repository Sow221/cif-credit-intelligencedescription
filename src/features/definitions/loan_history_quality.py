"""Contrat de la feature `loan_history_quality` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="loan_history_quality",
    version="1.0.0",
    family="history",
    description="Qualité globale de l'historique de prêts (0..1).",
    bounds=(0.0, 1.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
