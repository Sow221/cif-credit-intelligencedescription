"""Tests unitaires des contrats de features (exigence cabinet : un test par feature)."""

import pytest

from config.schema import DataConfig, FeatureConfig
from data.synthetic import generate_datasets
from features.builder import build_features
from features.definitions import CONTRACTS, FeatureContract, validate_feature_contracts


def _expected_features(cfg: FeatureConfig) -> set[str]:
    return {col for family in cfg.families.values() for col in family}


def test_every_feature_has_a_contract():
    cfg = FeatureConfig()
    expected = _expected_features(cfg)
    assert expected == set(CONTRACTS)
    assert len(CONTRACTS) == 25


def test_contract_metadata_are_complete():
    for name, contract in CONTRACTS.items():
        assert contract.name == name
        assert len(contract.version.split(".")) == 3
        assert contract.family in {"profile_income", "savings", "history", "context"}
        assert contract.description
        assert contract.owner in {"P1", "P2", "P3"}
        assert contract.sla
        lo, hi = contract.bounds
        assert lo is None or hi is None or lo <= hi


def test_validate_feature_contracts_returns_no_error():
    assert validate_feature_contracts() == []


def test_validate_feature_contracts_detects_orphan_contract():
    cfg = FeatureConfig()
    ghost = FeatureContract(
        name="ghost_feature",
        version="1.0.0",
        family="history",
        description="orphan",
        bounds=(0.0, 1.0),
        owner="P1",
        sla="n/a",
    )
    CONTRACTS["ghost_feature"] = ghost
    try:
        errors = validate_feature_contracts(cfg)
    finally:
        del CONTRACTS["ghost_feature"]
    assert any("orphelin" in e for e in errors)


def test_validate_bounds():
    contract = CONTRACTS["age"]
    assert contract.validate_bounds(30.0)
    assert not contract.validate_bounds(17.0)
    assert not contract.validate_bounds(90.0)


@pytest.mark.parametrize(
    "name,lo,hi",
    [
        ("age", 18.0, 75.0),
        ("monthly_income", 20_000.0, 2_000_000.0),
        ("n_past_loans", 0.0, 8.0),
        ("seniority_months", 0.0, 240.0),
        ("savings_volatility", 0.0, 1.0),
        ("savings_stability", 0.0, 1.0),
        ("avg_repayment_regularity", 0.0, 1.0),
        ("historical_default_rate", 0.0, 1.0),
        ("loan_to_savings_ratio", 0.0, 200.0),
    ],
)
def test_feature_bounds_respected_on_generated_data(name, lo, hi):
    cfg = FeatureConfig()
    ds = generate_datasets(DataConfig(n_customers=250, seed=11))
    features = build_features(ds.customers, ds.loans, ds.savings, cfg)
    values = features[name].dropna()
    assert values.between(lo, hi).all()


def test_feature_columns_are_stable():
    cfg = FeatureConfig()
    cols = [col for family in cfg.families.values() for col in family]
    assert len(cols) == len(set(cols))  # pas de doublon dans la config
