"""Contrat de données (Pandera) appliqué à la couche ``interim`` — bloquant en cas de violation."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from config.schema import LendingClubConfig
from data.lending_club import DATE_COL, POST_ORIGINATION_COLUMNS, TARGET, LendingClubDataError


def _schema(cfg: LendingClubConfig) -> pa.DataFrameSchema:
    start, end = pd.Timestamp(cfg.issue_start), pd.Timestamp(cfg.issue_end)
    return pa.DataFrameSchema(
        {
            "id": pa.Column(int, unique=True),
            DATE_COL: pa.Column("datetime64[ns]", pa.Check.in_range(start, end)),
            TARGET: pa.Column(int, pa.Check.isin([0, 1])),
            "loan_amnt": pa.Column(float, pa.Check.gt(0), coerce=True),
            "term": pa.Column(float, pa.Check.eq(float(cfg.term_months)), coerce=True),
            "annual_inc": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True, required=False),
            "dti": pa.Column(float, pa.Check.in_range(-1, 1000), nullable=True, coerce=True, required=False),
            "fico_range_low": pa.Column(float, pa.Check.in_range(300, 850), nullable=True, coerce=True, required=False),
            "revol_util": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True, required=False),
        },
        strict=False,
    )


def validate_clean(df: pd.DataFrame, cfg: LendingClubConfig) -> pd.DataFrame:
    """Valide le contrat ; lève ``LendingClubDataError`` avec le détail des violations."""
    leaked = sorted(set(df.columns) & POST_ORIGINATION_COLUMNS)
    if leaked:
        raise LendingClubDataError(f"Colonnes d'après-octroi présentes (fuite) : {leaked}")
    if df.empty:
        raise LendingClubDataError("Jeu vide après nettoyage : vérifiez le périmètre (dates, terme).")
    try:
        return _schema(cfg).validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise LendingClubDataError(f"Contrat de données violé :\n{exc.failure_cases.head(20)}") from exc
