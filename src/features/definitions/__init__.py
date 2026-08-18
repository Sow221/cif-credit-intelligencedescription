"""Registre des contrats de features (exigence cabinet : un fichier par feature).

Ce module agrège les 25 contrats et expose ``validate_feature_contracts`` qui garantit la
cohérence entre les contrats, la configuration (``FeatureConfig.families``) et la source de
vérité des colonnes du modèle — toute feature sans contrat ou tout
contrat orphelin est signalé.
"""

from __future__ import annotations

from config.schema import FeatureConfig
from features.definitions import (
    active_loans,
    age,
    avg_savings_24m,
    current_loan_duration,
    current_loan_purpose_num,
    current_loan_request,
    current_savings,
    debt_ratio,
    gender_num,
    income_to_request_ratio,
    income_volatility,
    loan_history_quality,
    loan_to_savings_ratio,
    location_num,
    max_dpd,
    monthly_income,
    n_past_loans,
    payments_on_time_ratio,
    repayment_regularity,
    savings_min_24m,
    savings_stability,
    savings_trend,
    savings_volatility,
    sector_num,
    seniority_months,
)
from features.definitions.contract import FeatureContract

_CONTRACT_MODULES = [
    age,
    active_loans,
    avg_savings_24m,
    current_loan_duration,
    current_loan_purpose_num,
    current_loan_request,
    current_savings,
    debt_ratio,
    gender_num,
    income_to_request_ratio,
    income_volatility,
    loan_history_quality,
    loan_to_savings_ratio,
    location_num,
    max_dpd,
    monthly_income,
    n_past_loans,
    payments_on_time_ratio,
    repayment_regularity,
    savings_min_24m,
    savings_stability,
    savings_trend,
    savings_volatility,
    sector_num,
    seniority_months,
]

CONTRACTS: dict[str, FeatureContract] = {module.CONTRACT.name: module.CONTRACT for module in _CONTRACT_MODULES}


def validate_feature_contracts(cfg: FeatureConfig | None = None) -> list[str]:
    """Vérifie l'intégrité des contrats vs la config et la source de vérité.

    Returns:
        Liste des erreurs de cohérence (vide si tout est conforme).
    """
    cfg = cfg or FeatureConfig()
    errors: list[str] = []

    expected = [col for family in cfg.families.values() for col in family]
    for name in expected:
        if name not in CONTRACTS:
            errors.append(f"feature {name!r} sans contrat dans features/definitions")
    for name in CONTRACTS:
        if name not in expected:
            errors.append(f"contrat orphelin {name!r} absent de FeatureConfig.families")

    for name, contract in CONTRACTS.items():
        if contract.name != name:
            errors.append(f"contrat {name!r} a un nom discordant {contract.name!r}")
        lo, hi = contract.bounds
        if lo is not None and hi is not None and lo > hi:
            errors.append(f"bornes invalides pour {name!r}: ({lo}, {hi})")

    return errors


__all__ = [
    "CONTRACTS",
    "FeatureContract",
    "validate_feature_contracts",
]
