"""Tests de la garde anti-leakage et du contrat des 25 features.

Verrouille le comportement défensif de la pipeline (porté depuis le dépôt de
référence `cifci`) : aucune variable de fuite ne doit entrer dans les features,
ni être émise par le générateur, ni être acceptée par l'API. Le jeu de features
doit être exactement de la taille annoncée (25 + id + cible).
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.schema import FeatureConfig
from features.builder import aggregate_loans, build_features
from features.validate import (
    FORBIDDEN_SUBSTRINGS,
    LeakageError,
    assert_no_correlation_leakage,
    assert_no_leakage,
    forbidden_features,
)


def test_forbidden_substrings_covered() -> None:
    for col in ["p_default_true", "p_default", "probability_default", "true_default"]:
        assert any(tok in col.lower() for tok in FORBIDDEN_SUBSTRINGS)
    assert "is_default" not in "".join(FORBIDDEN_SUBSTRINGS)


def test_forbidden_features_detects_p_default_true() -> None:
    df = pd.DataFrame({"a": [1, 2], "p_default_true": [0.1, 0.2]})
    assert forbidden_features(df) == ["p_default_true"]


def test_assert_no_leakage_raises_on_forbidden_column() -> None:
    df = pd.DataFrame({"a": [1], "p_default_true": [0.5]})
    with pytest.raises(LeakageError):
        assert_no_leakage(df)


def test_assert_no_leakage_ok_clean() -> None:
    df = pd.DataFrame({"a": [1, 2], "is_default": [0, 1], "customer_id": [1, 2]})
    out = assert_no_leakage(df, feature_columns=["a"], allow_target=True)
    assert list(out.columns) == ["a", "is_default", "customer_id"]


def test_build_features_raises_on_contaminated_customers() -> None:
    """Une table clients contenant `p_default_true` doit être bloquée d'emblée."""
    from config.schema import DataConfig
    from data.synthetic import generate_datasets

    ds = generate_datasets(DataConfig(n_customers=50, seed=3))
    ds.customers["p_default_true"] = 0.1
    with pytest.raises(LeakageError):
        build_features(ds.customers, ds.loans, ds.savings, FeatureConfig())


def test_assert_no_correlation_leakage_catches_structural_leak() -> None:
    """Une feature quasi-copie de la cible (corr > seuil) doit être bloquée, même sans
    nom suspect. Reproduit le bug historique : `historical_default_rate` calculé à partir
    d'un `loan_status` lui-même dérivé de `is_default` (corrélation mesurée ≈ 0.94)."""
    n = 500
    rng = pd.Series(range(n))
    target = (rng % 5 == 0).astype(int)
    almost_copy = target * 0.9 + 0.05  # corrélation quasi parfaite avec la cible
    df = pd.DataFrame({"honest_feature": rng % 7, "leaky_feature": almost_copy, "is_default": target})
    with pytest.raises(LeakageError):
        assert_no_correlation_leakage(df, feature_columns=["honest_feature", "leaky_feature"], target="is_default")


def test_assert_no_correlation_leakage_allows_honest_signal() -> None:
    df = pd.DataFrame({"a": [1, 5, 2, 4, 3, 3, 2, 4, 1, 5], "is_default": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]})
    out = assert_no_correlation_leakage(df, feature_columns=["a"], target="is_default")
    assert out is df


def test_build_features_generated_data_has_no_structural_leak() -> None:
    """Le générateur actuel (loan_status dérivé du comportement réalisé du prêt, jamais
    de `is_default`) ne doit produire aucune feature quasi-corrélée à la cible."""
    from config.schema import DataConfig
    from data.synthetic import generate_datasets

    ds = generate_datasets(DataConfig(n_customers=2000, seed=21))
    built = build_features(ds.customers, ds.loans, ds.savings, FeatureConfig())
    cols = [c for fam in FeatureConfig().families.values() for c in fam]
    corr = built[cols].astype(float).corrwith(built["is_default"].astype(float))
    assert corr.abs().max() < 0.75, f"Corrélation suspecte détectée : {corr.abs().idxmax()}={corr.abs().max():.3f}"


def test_build_features_generated_data_output_contract() -> None:
    """Sur le générateur propre, la sortie = 25 features + customer_id + cible."""
    from config.schema import DataConfig
    from data.synthetic import generate_datasets

    cfg = FeatureConfig()
    n_features = sum(len(fam) for fam in cfg.families.values())
    assert n_features == 25

    ds = generate_datasets(DataConfig(n_customers=60, seed=11))
    built = build_features(ds.customers, ds.loans, ds.savings, cfg)
    assert "p_default_true" not in built.columns
    # + customer_id + application_date (métadonnée de jointure point-in-time) + is_default
    assert len(built.columns) == n_features + 3


def test_aggregate_loans_excludes_loans_at_or_after_application_date() -> None:
    """Fuite temporelle : un prêt daté à ou après la demande courante d'un client ne doit
    JAMAIS entrer dans son agrégat d'historique — même si le générateur le garantit déjà
    par construction, cette garde doit le détecter pour de vraies données CIF (sans cette
    garantie) où un tel prêt existerait."""
    customers = pd.DataFrame({"customer_id": [1], "application_date": ["2025-01-01"]})
    loans = pd.DataFrame(
        {
            "customer_id": [1, 1, 1],
            "loan_amount": [100_000.0, 200_000.0, 300_000.0],
            "loan_start_date": ["2024-01-01", "2025-01-01", "2025-06-01"],  # avant / == / après
            "repayment_regularity": [0.9, 0.9, 0.9],
            "max_dpd": [0, 0, 0],
            "loan_status": ["repaid", "repaid", "repaid"],
        }
    )
    agg = aggregate_loans(loans, customers)
    row = agg[agg["customer_id"] == 1].iloc[0]
    # Seul le prêt du 2024-01-01 (strictement avant application_date) doit survivre.
    assert row["n_loans"] == 1
    assert row["total_loan_amount"] == 100_000.0


def test_aggregate_loans_without_customers_is_unfiltered() -> None:
    """Sans table customers (ou sans application_date), aggregate_loans reste permissif —
    utilisé par exemple pour des analyses hors pipeline d'entraînement."""
    loans = pd.DataFrame(
        {
            "customer_id": [1, 1],
            "loan_amount": [100_000.0, 200_000.0],
            "loan_start_date": ["2024-01-01", "2030-01-01"],
            "repayment_regularity": [0.9, 0.9],
            "max_dpd": [0, 0],
            "loan_status": ["repaid", "repaid"],
        }
    )
    agg = aggregate_loans(loans)
    assert agg[agg["customer_id"] == 1].iloc[0]["n_loans"] == 2
