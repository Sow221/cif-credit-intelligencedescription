"""Contrat de la feature `n_defaults` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="n_defaults",
    version="1.0.0",
    family="history",
    description="Nombre de prêts en défaut de paiement dans l'historique.",
    bounds=(0.0, None),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
