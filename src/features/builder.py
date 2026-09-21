"""Feature engineering versionné — 25 features du MODÈLE OFFICIEL.

Reconstruit le pipeline qui transforme les tables brutes ``customers`` +
``loans`` en la matrice des 25 features utilisées par le modèle officiel
calibré (``MODEL_OFFICIAL_CALIBRATED``). C'est LA source de vérité : cette
liste reproduit le ROC-AUC documenté (phase synthétique ≈ 0.83 / pipeline
reproduit ≈ 0.87). Elle est alignée sur le dépôt de référence `cifci`.

Familles (utilisées par l'ablation study) :
- profile_income : profil socio-démographique et revenu
- savings        : comportement d'épargne (profil)
- history        : historique de remboursement (agrégation des prêts)
- context        : contexte de la demande / ratios économiques

Garde anti-leakage : aucune variable de fuite (``p_default_true``…) ne peut
entrer — erreur bloquante si c'est le cas (``features.validate``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config.schema import FeatureConfig
from features.validate import (
    LeakageError,
    assert_no_correlation_leakage,
    assert_no_leakage,
    forbidden_features,
)
from utils.logging import get_logger

logger = get_logger(__name__)

# Colonnes de la table customers reprises telles quelles (id + cible gérés à part).
_CUSTOMER_FEATURE_COLS = [
    "age",
    "seniority_months",
    "monthly_income",
    "current_savings",
    "avg_savings_24m",
    "savings_std_24m",
    "savings_volatility",
    "savings_stability",
    "n_past_loans",
    "current_loan_request",
    "current_loan_duration",
    "loan_to_savings_ratio",
]


def aggregate_loans(loans: pd.DataFrame, customers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Agrège l'historique de prêts par client en features numériques (HISTORY).

    Jointure POINT-IN-TIME si ``customers`` fournit ``application_date`` : tout prêt daté à ou
    après la demande courante du client est exclu de l'agrégation. Le générateur synthétique
    garantit déjà cette propriété par construction (voir ``data/synthetic.py``), mais cette
    garde reste nécessaire : de vraies données CIF n'offriraient aucune telle garantie, et
    c'est exactement le genre de fuite temporelle qu'un simple split trié ne détecte pas.
    """
    loans = loans.copy()
    if customers is not None and "application_date" in customers.columns:
        cutoff = customers[["customer_id", "application_date"]].rename(columns={"application_date": "_as_of"})
        loans = loans.merge(cutoff, on="customer_id", how="left")
        loans = loans[pd.to_datetime(loans["loan_start_date"]) < pd.to_datetime(loans["_as_of"])]
        loans = loans.drop(columns=["_as_of"])
    agg = (
        loans.groupby("customer_id")
        .agg(
            n_loans=("loan_amount", "size"),
            avg_loan_amount=("loan_amount", "mean"),
            total_loan_amount=("loan_amount", "sum"),
            avg_repayment_regularity=("repayment_regularity", "mean"),
            min_repayment_regularity=("repayment_regularity", "min"),
            max_historical_dpd=("max_dpd", "max"),
            mean_historical_dpd=("max_dpd", "mean"),
            n_defaults=("loan_status", lambda s: int((s == "default").sum())),
        )
        .reset_index()
    )
    agg["historical_default_rate"] = agg["n_defaults"] / np.maximum(agg["n_loans"], 1)
    return agg


def build_features(
    customers: pd.DataFrame,
    loans: pd.DataFrame,
    savings: pd.DataFrame | None = None,
    cfg: FeatureConfig | None = None,
    *,
    drop_forbidden: bool = False,
) -> pd.DataFrame:
    """Construit la matrice des 25 features officielles (+ ``customer_id`` + cible).

    Args:
        customers: table clients (avec ``customer_id``, ``is_default``).
        loans: historique de prêts à agréger par client.
        savings: ignoré — les 25 features officielles ne dépendent que de
            customers + loans. Conservé pour compatibilité d'appel.
        cfg: configuration des features (familles / cible).
        drop_forbidden: si True, supprime en avertissant toute variable de fuite
            présente en entrée ; sinon (défaut), lève ``LeakageError`` — garde
            déterministe, aucune fuite ne survit.

    Returns:
        DataFrame des 25 features officielles + ``customer_id`` + ``is_default``.
    """
    cfg = cfg or FeatureConfig()
    feature_columns = [col for family in cfg.families.values() for col in family]
    target = cfg.target

    hits = forbidden_features(customers)
    if hits:
        if drop_forbidden:
            warnings.warn(
                f"Variables de fuite détectées et supprimées : {hits}.",
                stacklevel=2,
            )
            customers = customers.drop(columns=[c for c in hits if c in customers])
        else:
            raise LeakageError(
                f"Variables de fuite détectées en entrée (customers) : {hits}. Supprimez-les avant tout entraînement."
            )

    loan_agg = aggregate_loans(loans, customers)
    df = customers.merge(loan_agg, on="customer_id", how="left")

    # Remplit les clients sans historique (thin-file) par des neutres.
    for col in [
        "n_loans",
        "avg_loan_amount",
        "total_loan_amount",
        "avg_repayment_regularity",
        "min_repayment_regularity",
        "max_historical_dpd",
        "mean_historical_dpd",
        "n_defaults",
        "historical_default_rate",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Features dérivées (ratios et conversions d'unités).
    df["loan_to_income_ratio"] = df["current_loan_request"] / df["monthly_income"].replace(0, np.nan)
    df["savings_to_income_ratio"] = df["current_savings"] / df["monthly_income"].replace(0, np.nan)
    df["seniority_years"] = df["seniority_months"] / 12.0
    df["overall_payment_regularity"] = df["avg_repayment_regularity"]

    out_cols = ["customer_id", *feature_columns]
    if "application_date" in df.columns:
        out_cols.append("application_date")
    if target in df.columns:
        out_cols.append(target)
    result = df[out_cols].copy()

    result = assert_no_leakage(result, feature_columns=feature_columns, allow_target=True)
    if target in result.columns:
        result = assert_no_correlation_leakage(result, feature_columns=feature_columns, target=target)
    return result


def save_features(df: pd.DataFrame, cfg: FeatureConfig, processed_dir: str) -> str:
    """Persiste le jeu de features final en Parquet."""
    from pathlib import Path

    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / cfg.output_file
    df.to_parquet(path, index=False)
    return str(path)
