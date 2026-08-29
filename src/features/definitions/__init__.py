"""Registre des contrats de features (exigence cabinet : un fichier par feature).

Ce module agrège les 25 contrats — les features du MODÈLE OFFICIEL calibré — et
expose ``validate_feature_contracts`` qui garantit la cohérence entre les
contrats, la configuration (``FeatureConfig.families``) et la source de vérité
des colonnes du modèle — toute feature sans contrat ou tout contrat orphelin
est signalé.
"""

from __future__ import annotations

from config.schema import FeatureConfig
from features.definitions import (
    age,
    avg_loan_amount,
    avg_repayment_regularity,
    avg_savings_24m,
    current_loan_duration,
    current_loan_request,
    current_savings,
    historical_default_rate,
    loan_to_income_ratio,
    loan_to_savings_ratio,
    max_historical_dpd,
    mean_historical_dpd,
    min_repayment_regularity,
    monthly_income,
    n_defaults,
    n_loans,
    n_past_loans,
    overall_payment_regularity,
    savings_stability,
    savings_std_24m,
    savings_to_income_ratio,
    savings_volatility,
    seniority_months,
    seniority_years,
    total_loan_amount,
)
from features.definitions.contract import FeatureContract

_CONTRACT_MODULES = [
    age,
    seniority_months,
    monthly_income,
    current_loan_request,
    current_loan_duration,
    current_savings,
    avg_savings_24m,
    savings_std_24m,
    savings_volatility,
    savings_stability,
    loan_to_savings_ratio,
    savings_to_income_ratio,
    n_past_loans,
    n_loans,
    avg_loan_amount,
    total_loan_amount,
    avg_repayment_regularity,
    min_repayment_regularity,
    max_historical_dpd,
    mean_historical_dpd,
    n_defaults,
    historical_default_rate,
    overall_payment_regularity,
    loan_to_income_ratio,
    seniority_years,
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
