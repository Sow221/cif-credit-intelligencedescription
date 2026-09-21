"""Ingestion du jeu public Lending Club : lecture, nettoyage, label mature, garde anti-fuite.

Principes (standards du scoring de crédit) :

* **Point-in-time** : seules des colonnes connues *à l'octroi* sont conservées. Les colonnes
  d'après-octroi (``total_pymnt``, ``out_prncp``, ``recoveries``, ``last_pymnt_d``…) sont
  interdites : elles encodent l'issue du prêt.
* **Label mature** : seuls les prêts terminés (remboursés / passés en perte) sont gardés, sur
  un périmètre où tous les prêts ont eu le temps d'arriver à terme (voir ``LendingClubConfig``).
* **Données brutes immuables** : ce module lit ``data/raw`` et n'écrit que dans ``data/interim``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.schema import LendingClubConfig
from utils.logging import get_logger

logger = get_logger(__name__)

TARGET = "is_default"
DATE_COL = "issue_d"

# Colonnes connues à la demande de crédit (point-in-time) + identifiant, date et statut.
APPLICATION_COLUMNS: tuple[str, ...] = (
    "id",
    "loan_amnt",
    "term",
    "installment",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
)
LENDER_SCORE_COLUMNS: tuple[str, ...] = ("grade", "sub_grade", "int_rate")

# Colonnes qui n'existent qu'après l'octroi : leur présence dans les features est une fuite.
POST_ORIGINATION_COLUMNS: frozenset[str] = frozenset(
    {
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
        "hardship_flag",
        "debt_settlement_flag",
        "settlement_status",
        "funded_amnt",
        "funded_amnt_inv",
    }
)

_DEFAULT_STATUSES = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
_PAID_STATUSES = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}


class LendingClubDataError(Exception):
    """Levée quand le fichier source est absent ou viole le contrat de données."""


def find_raw_file(raw_dir: str | Path) -> Path:
    """Localise le fichier des prêts acceptés (l'archive Kaggle peut être imbriquée)."""
    root = Path(raw_dir)
    candidates = sorted(p for p in root.rglob("*") if p.is_file() and p.name.lower().startswith("accepted"))
    candidates += sorted(p for p in root.rglob("*.csv*") if p.is_file() and p not in candidates)
    candidates = [p for p in candidates if "reject" not in p.name.lower()]
    if not candidates:
        raise LendingClubDataError(
            f"Aucun fichier Lending Club dans {root}. Lancez `make data-download` (voir data/README.md)."
        )
    return candidates[0]


def load_raw(path: str | Path, cfg: LendingClubConfig, nrows: int | None = None) -> pd.DataFrame:
    """Lit uniquement les colonnes utiles (le fichier complet dépasse 1 Go)."""
    wanted = set(APPLICATION_COLUMNS) | set(LENDER_SCORE_COLUMNS)
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False, nrows=nrows)
    missing = {"id", "loan_status", DATE_COL, "loan_amnt", "term"} - set(df.columns)
    if missing:
        raise LendingClubDataError(f"Colonnes obligatoires absentes du fichier source : {sorted(missing)}")
    logger.info("data.lending_club.loaded", rows=len(df), path=str(path))
    return df


def _parse_month_year(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")


def _parse_emp_length(s: pd.Series) -> pd.Series:
    text = s.astype("string").str.strip()
    years = text.str.extract(r"(\d+)")[0].astype("Float64")
    years = years.mask(text.str.startswith("<", na=False), 0.0)
    return years.astype(float)


def clean(raw: pd.DataFrame, cfg: LendingClubConfig) -> pd.DataFrame:
    """Nettoie, dérive le label mature et restreint au périmètre configuré."""
    # Liste blanche : toute colonne hors « connu à l'octroi » est écartée, quelle que soit la source.
    allowed = set(APPLICATION_COLUMNS) | set(LENDER_SCORE_COLUMNS)
    df = raw[[c for c in raw.columns if c in allowed]].copy()

    df[DATE_COL] = _parse_month_year(df[DATE_COL])
    df = df.dropna(subset=[DATE_COL, "loan_status"])

    status = df["loan_status"].astype(str).str.strip()
    df[TARGET] = np.select([status.isin(_DEFAULT_STATUSES), status.isin(_PAID_STATUSES)], [1, 0], default=-1)
    df = df[df[TARGET] >= 0]
    df = df.drop(columns=["loan_status"])

    df["term"] = df["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    df = df[df["term"] == cfg.term_months]

    start, end = pd.Timestamp(cfg.issue_start), pd.Timestamp(cfg.issue_end)
    df = df[(df[DATE_COL] >= start) & (df[DATE_COL] <= end)]

    if "emp_length" in df:
        df["emp_length"] = _parse_emp_length(df["emp_length"])
    if "earliest_cr_line" in df:
        df["earliest_cr_line"] = _parse_month_year(df["earliest_cr_line"])
    if "revol_util" in df and df["revol_util"].dtype == object:
        df["revol_util"] = pd.to_numeric(df["revol_util"].astype(str).str.rstrip("%"), errors="coerce")

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).drop_duplicates(subset="id")
    df["id"] = df["id"].astype("int64")

    if cfg.exclude_lender_score:
        df = df.drop(columns=[c for c in LENDER_SCORE_COLUMNS if c in df.columns])

    df = df.sort_values([DATE_COL, "id"]).reset_index(drop=True)
    logger.info(
        "data.lending_club.cleaned",
        rows=len(df),
        default_rate=round(float(df[TARGET].mean()), 4) if len(df) else None,
        first=str(df[DATE_COL].min()),
        last=str(df[DATE_COL].max()),
    )
    return df


def build_interim(cfg: LendingClubConfig, nrows: int | None = None) -> Path:
    """Chaîne complète raw → interim (validé) ; retourne le chemin du parquet nettoyé."""
    from data.validation import validate_clean

    raw = load_raw(find_raw_file(cfg.raw_dir), cfg, nrows=nrows)
    df = validate_clean(clean(raw, cfg), cfg)
    out = Path(cfg.interim_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("data.lending_club.interim_saved", path=str(out), rows=len(df))
    return out


def temporal_partition(
    df: pd.DataFrame, cfg: LendingClubConfig, date_col: str = DATE_COL
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe train / validation / test par vraie date d'octroi (jamais aléatoire)."""
    train_end, valid_end = pd.Timestamp(cfg.train_end), pd.Timestamp(cfg.valid_end)
    if not train_end < valid_end:
        raise ValueError("train_end doit précéder valid_end")
    train = df[df[date_col] <= train_end]
    valid = df[(df[date_col] > train_end) & (df[date_col] <= valid_end)]
    test = df[df[date_col] > valid_end]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f"Partition vide : train={len(train)} valid={len(valid)} test={len(test)}")
    return train, valid, test
