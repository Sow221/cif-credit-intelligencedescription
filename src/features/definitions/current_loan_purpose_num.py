"""Contrat de la feature `current_loan_purpose_num` (famille context).

Generated — voir features/definitions/contract.py.
"""

from __future__ import annotations

from features.definitions.contract import FeatureContract

CONTRACT = FeatureContract(
    name="current_loan_purpose_num",
    version="1.0.0",
    family="context",
    description="Objet du prêt courant encodé.",
    bounds=(-1.0, 3.0),
    owner="P1",
    sla="disponible avant décision - batch quotidien",
)
