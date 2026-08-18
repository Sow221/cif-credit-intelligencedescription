"""Feature engineering versionné — 25 features organisées en 4 familles.

Familles (alignées avec l'ablation study du journal de bord) :
- profile_income : profil socio-démographique et revenu
- savings        : comportement d'épargne (signal dominant)
- history        : historique de remboursement
- context        : contexte de la demande courante

Toutes les features sont calculées à partir d'informations **disponibles avant la décision**
(principe anti-leakage du protocole CIF : aucune variable post-décision).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cif_credit.config.schema import FeatureConfig
from cif_credit.utils.logging import get_logger

logger = get_logger(__name__)

SECTOR_MAP = {"agriculture": 0, "commerce": 1, "services": 2, "elevage": 3}
LOCATION_MAP = {"urbain": 0, "periurbain": 1, "rural": 2}
PURPOSE_MAP = {"agricole": 0, "commercial": 1, "equipement": 2, "consommation": 3}


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les variables catégorielles en codes numériques stables (pas de one-hot pour limiter la dim.)."""
    out = df.copy()
    out["gender_num"] = (out["gender"] == "F").astype(int)
    out["sector_num"] = out["sector"].map(SECTOR_MAP).fillna(-1).astype(int)
    out["location_num"] = out["location"].map(LOCATION_MAP).fillna(-1).astype(int)
    out["current_loan_purpose_num"] = out["current_loan_purpose"].map(PURPOSE_MAP).fillna(-1).astype(int)
    return out


def aggregate_loans(customers: pd.DataFrame, loans: pd.DataFrame) -> pd.DataFrame:
    """Agrège l'historique de prêts par client (features HISTORY).

    Le comptage est nommé ``n_past_loans_history`` pour éviter toute collision
    avec la colonne ``n_past_loans`` de la table customers.
    """
    agg = (
        loans.groupby("customer_id")
        .agg(
            n_past_loans_history=("loan_id", "count"),
            repayment_regularity=("repayment_regularity", "mean"),
            max_dpd=("max_dpd", "max"),
            payments_on_time_ratio=("payments_on_time", "mean"),
            n_payments_total=("n_payments", "sum"),
            active_loans=("loan_status", lambda s: int((s == "open").sum())),
        )
        .reset_index()
    )
    # qualité globale de l'historique (0..1)
    agg["loan_history_quality"] = np.clip(
        agg["repayment_regularity"]
        * (1.0 - agg["max_dpd"] / 90.0)
        * (agg["payments_on_time_ratio"].clip(0, 1)),
        0.0,
        1.0,
    )
    return agg


def aggregate_savings(customers: pd.DataFrame, savings: pd.DataFrame) -> pd.DataFrame:
    """Agrège les séries temporelles d'épargne (features SAVINGS dérivées)."""
    agg = (
        savings.groupby("customer_id")
        .agg(
            savings_trend_raw=("savings_balance", lambda s: np.polyfit(np.arange(len(s)), s.values, 1)[0]),
            savings_min_24m=("savings_balance", "min"),
        )
        .reset_index()
    )
    return agg


def build_features(
    customers: pd.DataFrame, loans: pd.DataFrame, savings: pd.DataFrame, cfg: FeatureConfig
) -> pd.DataFrame:
    """Construit le jeu de features final (25 colonnes + target)."""
    df = customers.copy()
    df = encode_categorical(df)

    loan_agg = aggregate_loans(customers, loans)
    savings_agg = aggregate_savings(customers, savings)
    df = df.merge(loan_agg, on="customer_id", how="left")
    df = df.merge(savings_agg, on="customer_id", how="left")

    # Combler les clients sans historique (thin-file) ; le comptage issu de l'historique
    # réel des prêts prime sur la déclaration de la table customers.
    df["n_past_loans"] = df["n_past_loans_history"].fillna(df["n_past_loans"]).fillna(0).astype(int)
    df = df.drop(columns=["n_past_loans_history"])
    df["repayment_regularity"] = df["repayment_regularity"].fillna(0.5)
    df["max_dpd"] = df["max_dpd"].fillna(0).astype(int)
    df["payments_on_time_ratio"] = df["payments_on_time_ratio"].fillna(0.5)
    df["loan_history_quality"] = df["loan_history_quality"].fillna(0.0)
    df["active_loans"] = df["active_loans"].fillna(0).astype(int)

    # Features dérivées supplémentaires
    df["savings_trend"] = df["savings_trend_raw"].fillna(0.0)
    df["savings_min_24m"] = df["savings_min_24m"].fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["income_to_request_ratio"] = np.where(
            df["current_loan_request"] > 0,
            df["monthly_income"] / df["current_loan_request"],
            0.0,
        )
        df["debt_ratio"] = np.where(
            df["monthly_income"] > 0,
            df["current_loan_request"] / np.maximum(df["monthly_income"] * 12.0, 1.0),
            0.0,
        )
    df["income_to_request_ratio"] = df["income_to_request_ratio"].replace([np.inf, -np.inf], 0.0).clip(0, 10)
    df["debt_ratio"] = df["debt_ratio"].replace([np.inf, -np.inf], 0.0).clip(0, 20)

    df = df.drop(columns=["savings_trend_raw"], errors="ignore")

    # Sélection finale (ordre stable et reproductible)
    feature_cols: list[str] = []
    for family in cfg.families.values():
        for col in family:
            if col not in feature_cols:
                feature_cols.append(col)
    required = [*feature_cols, cfg.target]

    # Le customer_id est conservé pour le traçage individuel des décisions (audit §M08),
    # mais ne fait PAS partie des features du modèle.
    if "customer_id" in df.columns and "customer_id" not in required:
        required = ["customer_id", *required]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes après feature engineering : {missing}")

    logger.info(
        "features.build.done",
        n_features=len(feature_cols),
        n_rows=len(df),
        target=cfg.target,
    )
    return df[required].copy()


def save_features(df: pd.DataFrame, cfg: FeatureConfig, processed_dir: str) -> str:
    """Persiste le jeu de features final en Parquet."""
    from pathlib import Path

    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / cfg.output_file
    df.to_parquet(path, index=False)
    return str(path)
