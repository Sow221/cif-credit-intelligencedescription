"""Tests du générateur de données synthétiques."""

from config.schema import DataConfig
from data.synthetic import generate_datasets


def test_generate_datasets_structure():
    cfg = DataConfig(n_customers=500, n_savings_months=24, seed=7)
    ds = generate_datasets(cfg)

    assert len(ds.customers) == 500
    assert set(ds.customers["is_default"]).issubset({0, 1})
    # Garde anti-leakage à la source : la probabilité de fuite n'est JAMAIS émise
    # dans la table clients (elle servirait de variable de fuite à l'entraînement).
    assert "p_default_true" not in ds.customers.columns

    assert len(ds.loans) > 0
    assert set(ds.loans.columns) >= {
        "customer_id",
        "loan_id",
        "loan_amount",
        "loan_status",
        "max_dpd",
        "repayment_regularity",
    }

    assert len(ds.savings) == 500 * 24
    assert ds.savings["month"].min() == 1
    assert ds.savings["month"].max() == 24


def test_default_rate_calibration():
    cfg = DataConfig(n_customers=2000, default_rate=0.1183, seed=42)
    ds = generate_datasets(cfg)
    observed = ds.customers["is_default"].mean()
    # Tolérance large pour l'aléa d'échantillonnage
    assert abs(observed - cfg.default_rate) < 0.02


def test_reproducibility():
    cfg = DataConfig(n_customers=300, seed=99)
    a = generate_datasets(cfg)
    b = generate_datasets(cfg)
    assert a.customers.equals(b.customers)
    assert a.loans.equals(b.loans)
    assert a.savings.equals(b.savings)


def test_p_default_never_leaks_into_customers():
    """Le générateur ne doit exposer aucune variable de fuite dans la table clients.

    Autrefois `p_default_true` (probabilité de défaut exacte) était émis dans la
    table clients : c'est une variable de fuite catégorique — elle encode la cible
    par construction. Purge à la source (voir src/data/synthetic.py).
    """
    cfg = DataConfig(n_customers=1000, seed=1)
    ds = generate_datasets(cfg)
    lowered = {str(c).lower() for c in ds.customers.columns}
    assert all(not any(t in c for t in ("p_default", "probability_default", "true_default")) for c in lowered)
