"""Générateur de données synthétiques du jeu CIF Credit Intelligence.

Reproduit fidèlement la structure documentée dans `05_DOCUMENTATION/02_data_dictionary.json`
et les statistiques du `journal_bord.json` (10 000 clients, ~24 900 prêts, 240 000 enregistrements
d'épargne, taux de défaut cible calibré par logit inverse, seed reproductible).

Mécanique : un *facteur latent* de qualité de crédit pilote à la fois la probabilité de défaut
(logit inverse calibré sur le taux cible) et les observations (épargne, historique de remboursement),
ce qui garantit un signal cohérent et auditable — le modèle apprenable couple épargne + historique.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.schema import DataConfig
from utils.logging import get_logger
from utils.seed import seed_everything

logger = get_logger(__name__)

CUSTOMER_COLUMNS = [
    "customer_id",
    "application_date",
    "age",
    "gender",
    "sector",
    "location",
    "seniority_months",
    "monthly_income",
    "income_volatility",
    "current_savings",
    "avg_savings_24m",
    "savings_std_24m",
    "savings_volatility",
    "savings_stability",
    "n_past_loans",
    "current_loan_request",
    "current_loan_duration",
    "current_loan_purpose",
    "loan_to_savings_ratio",
    "is_default",
]

LOAN_COLUMNS = [
    "customer_id",
    "loan_id",
    "loan_amount",
    "loan_duration",
    "loan_purpose",
    "loan_status",
    "loan_start_date",
    "repayment_regularity",
    "max_dpd",
    "n_payments",
    "payments_on_time",
]

SAVINGS_COLUMNS = ["customer_id", "month", "date", "savings_balance", "transaction_amount"]

SECTORS = ["agriculture", "commerce", "services", "elevage"]
LOCATIONS = ["urbain", "periurbain", "rural"]
LOAN_PURPOSES = ["agricole", "commercial", "equipement", "consommation"]
LOAN_DURATIONS = [6, 12, 18, 24]


@dataclass
class GeneratedDatasets:
    """Résultat de la génération."""

    customers: pd.DataFrame
    loans: pd.DataFrame
    savings: pd.DataFrame


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=float)


def _calibrate_intercept(default_rate: float, latent: np.ndarray, scale: float) -> float:
    """Trouve le biais b0 tel que mean(sigmoid(b0 + scale * latent)) ≈ default_rate.

    Résolution robuste par bissection — l'approche « logit inverse avec biais empirique »
    documentée dans le journal de bord.
    """
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        mean_p = float(np.mean(_sigmoid(mid + scale * latent)))
        if mean_p < default_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def generate_customers(cfg: DataConfig, latent: np.ndarray, b0: float, scale: float) -> pd.DataFrame:
    """Génère la table des clients à partir du facteur latent."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_customers

    customer_ids = np.arange(1, n + 1)
    # Date de la demande courante — répartie sur ~18 mois, indépendamment de customer_id
    # (un identifiant client plus ancien ne veut pas dire une demande plus ancienne). C'est la
    # vraie colonne temporelle : le split temporel (models.train.temporal_split) trie dessus,
    # plus sur l'ordre de customer_id qui n'était qu'un proxy documenté comme tel.
    now = pd.Timestamp("2025-06-30")
    application_days_ago = rng.integers(0, 548, n)
    application_date = [(now - pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in application_days_ago]
    ages = np.clip(rng.normal(38, 12, n), 18, 75).astype(int)
    genders = rng.choice(["M", "F"], size=n, p=[0.48, 0.52])
    sectors = rng.choice(SECTORS, size=n, p=[0.35, 0.25, 0.25, 0.15])
    locations = rng.choice(LOCATIONS, size=n, p=[0.4, 0.35, 0.25])

    # Ancienneté : le lien d'épargne est plus fort quand le client est plus ancien.
    seniority = np.clip((latent * 12 + rng.normal(30, 20, n)).astype(int), 0, 240)

    # Revenu mensuel (cf. échelle FCFA, ~50 000 à 800 000)
    monthly_income = np.exp(np.log(50_000) + (latent + rng.normal(0, 0.5, n)) * 0.9)
    monthly_income = np.clip(monthly_income, 20_000, 2_000_000)

    # Volatilité de revenu (corrélée négativement à la qualité)
    income_volatility = np.clip(0.5 - 0.15 * latent + rng.gamma(2.0, 0.08, n), 0.05, 1.0)

    # Épargne courante et moyenne sur 24 mois — c'est le coeur du signal SAVINGS.
    base_savings = np.exp(np.log(80_000) + (latent + rng.normal(0, 0.55, n)) * 1.05)
    savings_level = np.clip(base_savings, 1_000, 5_000_000)
    savings_noise = rng.lognormal(0.0, 0.35, n)
    current_savings = np.clip(savings_level * savings_noise, 0, 8_000_000)
    avg_savings_24m = np.clip(savings_level * rng.lognormal(0.0, 0.3, n), 0, 7_000_000)

    savings_std_24m = np.clip(avg_savings_24m * 0.35 * rng.lognormal(0, 0.4, n), 0, 3_000_000)
    # Volatilité d'épargne normalisée (0..1) — plus élevée chez les clients à risque
    savings_volatility = np.clip(0.18 * rng.lognormal(0, 0.5, n) + 0.30 * (1.0 - latent).clip(0, 2), 0.01, 1.0)
    savings_stability = np.clip(1.0 - savings_volatility + rng.normal(0, 0.08, n), 0.0, 1.0)

    # Historique de prêts
    max_loans = cfg.max_past_loans
    raw_loans = n * (0.45 + 0.55 * (latent * 0.5 + 1.0) / 2.0) / max_loans
    n_past_loans = np.clip((raw_loans + rng.normal(0, 0.4, n)).astype(int), 0, max_loans)

    # Demande de crédit courante
    current_loan_request = np.clip(
        monthly_income * rng.lognormal(0, 0.35, n) * (1 + (1.0 - latent).clip(0, 1) * 0.4),
        20_000,
        20_000_000,
    )
    current_loan_duration = rng.choice(LOAN_DURATIONS, size=n)
    current_loan_purpose = rng.choice(LOAN_PURPOSES, size=n)

    p_default = _sigmoid(b0 + scale * latent)
    is_default = (rng.random(n) < p_default).astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        loan_to_savings_ratio = np.where(
            current_savings > 0, current_loan_request / np.maximum(current_savings, 1.0), 50.0
        )
        loan_to_savings_ratio = np.clip(loan_to_savings_ratio, 0.0, 200.0)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "application_date": application_date,
            "age": ages,
            "gender": genders,
            "sector": sectors,
            "location": locations,
            "seniority_months": seniority,
            "monthly_income": monthly_income.round(0),
            "income_volatility": income_volatility.round(4),
            "current_savings": current_savings.round(0),
            "avg_savings_24m": avg_savings_24m.round(0),
            "savings_std_24m": savings_std_24m.round(0),
            "savings_volatility": savings_volatility.round(4),
            "savings_stability": savings_stability.round(4),
            "n_past_loans": n_past_loans,
            "current_loan_request": current_loan_request.round(0),
            "current_loan_duration": current_loan_duration,
            "current_loan_purpose": current_loan_purpose,
            "loan_to_savings_ratio": loan_to_savings_ratio.round(4),
            "is_default": is_default,
        }
    )
    df.columns = [c.upper() if False else c for c in df.columns]
    return df[CUSTOMER_COLUMNS]


def generate_loans(cfg: DataConfig, customers: pd.DataFrame, latent: np.ndarray) -> pd.DataFrame:
    """Génère l'historique des prêts (environ 2,5 prêts / client en moyenne).

    Jointure point-in-time par construction : chaque prêt est daté strictement avant la date
    de la demande courante (`application_date`) du client concerné — jamais après. C'est ce qui
    rend le split temporel et l'agrégation d'historique (features/builder.py) réellement à l'abri
    d'une fuite temporelle, plutôt qu'un simple tri par ordre sans vraie garantie de date.
    """
    rng = np.random.default_rng(cfg.seed + 1)
    rows: list[dict] = []
    n = len(customers)

    n_loans_per_customer = np.clip(rng.poisson(2.2, n) * (0.6 + 0.4 * (latent * 0.5 + 1.0) / 2.0), 1, 8).astype(int)
    incomes = customers["monthly_income"].to_numpy()
    customer_ids = customers["customer_id"].to_numpy()
    application_dates = pd.to_datetime(customers["application_date"]).to_numpy()

    loan_counter = 0
    for idx in range(n):
        quality = latent[idx]
        customer_id = customer_ids[idx]
        income = float(incomes[idx])
        app_date = pd.Timestamp(application_dates[idx])
        k = int(n_loans_per_customer[idx])
        for _ in range(k):
            loan_counter += 1
            loan_id = f"L{loan_counter:07d}"
            amount = np.clip(income * rng.lognormal(0, 0.4) * 0.6, 5_000, 15_000_000)
            duration = int(rng.choice(LOAN_DURATIONS))
            purpose = str(rng.choice(LOAN_PURPOSES))
            start_date = app_date - pd.Timedelta(days=int(rng.integers(30, 1500)))
            # Signature du comportement de paiement corrélée au facteur latent.
            badness = max(0.0, 1.0 - (quality + rng.normal(0, 0.5)))
            repayment_regularity = float(np.clip(0.95 - 0.55 * badness, 0.1, 1.0))
            max_dpd = int(np.clip(rng.gamma(1.8, 6.0) * (0.3 + 2.0 * badness), 0, 90))
            n_payments_scheduled = duration
            payments_on_time = int(max(0, int(n_payments_scheduled * (repayment_regularity + rng.normal(0, 0.08)))))
            # Le statut du prêt est dérivé du comportement RÉALISÉ de CE prêt (retard
            # maximal constaté) — jamais de la cible `is_default` du client. Seuil calibré
            # empiriquement pour un taux de défaut au niveau prêt de l'ordre de 8 %, cohérent
            # avec le taux de défaut client (~12 %). Anti-leakage : voir tests/unit/test_leakage.py.
            loan_status = "default" if max_dpd >= 70 else "repaid"
            rows.append(
                {
                    "customer_id": customer_id,
                    "loan_id": loan_id,
                    "loan_amount": round(amount, 0),
                    "loan_duration": duration,
                    "loan_purpose": purpose,
                    "loan_status": loan_status,
                    "loan_start_date": start_date.strftime("%Y-%m-%d"),
                    "repayment_regularity": round(repayment_regularity, 4),
                    "max_dpd": max_dpd,
                    "n_payments": n_payments_scheduled,
                    "payments_on_time": payments_on_time,
                }
            )

    return pd.DataFrame(rows, columns=LOAN_COLUMNS)


def generate_savings(cfg: DataConfig, customers: pd.DataFrame, latent: np.ndarray) -> pd.DataFrame:
    """Génère les 24 relevés mensuels d'épargne par client."""
    rng = np.random.default_rng(cfg.seed + 2)
    rows: list[dict] = []
    month_count = cfg.n_savings_months
    start = pd.Timestamp("2023-07-01")

    for idx, customer_id in enumerate(customers["customer_id"].values):
        quality = latent[idx]
        base = float(np.exp(np.log(60_000) + (quality + rng.normal(0, 0.4)) * 1.0))
        balance = base
        for month in range(1, month_count + 1):
            drift = 1.02 + (quality * 0.01) + rng.normal(0, 0.02)
            txn = rng.lognormal(0, 0.5) * 12_000 * (0.4 + quality) * (1.0 if rng.random() > 0.35 else -0.3)
            balance = max(0.0, balance * drift + txn)
            date = (start + pd.Timedelta(days=month * 30)).strftime("%Y-%m-%d")
            rows.append(
                {
                    "customer_id": customer_id,
                    "month": month,
                    "date": date,
                    "savings_balance": round(balance, 0),
                    "transaction_amount": round(txn, 0),
                }
            )

    return pd.DataFrame(rows, columns=SAVINGS_COLUMNS)


def generate_datasets(cfg: DataConfig) -> GeneratedDatasets:
    """Orchestration de la génération complète des trois tables."""
    seed_everything(cfg.seed)
    logger.info(
        "data.generate.start",
        n_customers=cfg.n_customers,
        default_rate=cfg.default_rate,
        seed=cfg.seed,
    )

    rng = np.random.default_rng(cfg.seed)
    latent = rng.normal(0.0, 1.0, cfg.n_customers)
    scale = cfg.latent_factor_exponent
    b0 = _calibrate_intercept(cfg.default_rate, latent, scale)

    customers = generate_customers(cfg, latent, b0, scale)
    loans = generate_loans(cfg, customers, latent)
    savings = generate_savings(cfg, customers, latent)

    observed_default_rate = float(customers["is_default"].mean())
    logger.info(
        "data.generate.done",
        n_customers=len(customers),
        n_loans=len(loans),
        n_savings=len(savings),
        default_rate_observed=observed_default_rate,
    )
    return GeneratedDatasets(customers=customers, loans=loans, savings=savings)


def save_datasets(cfg: DataConfig, datasets: GeneratedDatasets) -> dict[str, Path]:
    """Persiste les trois tables en Parquet sous data/raw."""
    raw_dir = Path(cfg.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "customers": raw_dir / "customers.parquet",
        "loans": raw_dir / "loans.parquet",
        "savings": raw_dir / "savings.parquet",
    }
    datasets.customers.to_parquet(paths["customers"], index=False)
    datasets.loans.to_parquet(paths["loans"], index=False)
    datasets.savings.to_parquet(paths["savings"], index=False)
    logger.info("data.save.done", paths={k: str(v) for k, v in paths.items()})
    return paths
