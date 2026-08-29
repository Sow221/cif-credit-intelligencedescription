"""Contrat de la feature `overall_payment_regularity` (famille history).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="overall_payment_regularity",
    version="1.0.0",
    family="history",
    description="Régularité globale de paiement du client (0..1).",
    bounds=(0.0, 1.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
