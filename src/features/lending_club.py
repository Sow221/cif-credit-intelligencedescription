"""Features Lending Club — calculées à la date d'octroi (point-in-time), sans fuite.

Toutes les entrées sont connues au moment de la demande. L'ancienneté de crédit est calculée
*relativement à la date d'octroi* (et non à la date du jour), sinon elle varierait avec la date
d'exécution du pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.lending_club import DATE_COL, POST_ORIGINATION_COLUMNS, TARGET

NUMERIC_FEATURES: tuple[str, ...] = (
    "loan_amnt",
    "installment",
    "emp_length",
    "annual_inc_log",
    "dti",
    "delinq_2yrs",
    "credit_history_years",
    "fico_mean",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal_log",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "loan_to_income",
    "installment_to_income",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("home_ownership", "verification_status", "purpose", "application_type")
FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Sens économique attendu (contraintes de monotonicité) : +1 = le risque ne peut que monter.
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "fico_mean": -1,
    "annual_inc_log": -1,
    "dti": 1,
    "delinq_2yrs": 1,
    "inq_last_6mths": 1,
    "pub_rec": 1,
    "revol_util": 1,
    "loan_to_income": 1,
    "installment_to_income": 1,
}


def build_features(interim: pd.DataFrame) -> pd.DataFrame:
    """Construit la matrice de features + id, date et cible."""
    leaked = sorted(set(FEATURES) & POST_ORIGINATION_COLUMNS)
    if leaked:  # garde structurelle : la liste de features ne doit jamais contenir de colonne d'après-octroi
        raise ValueError(f"Fuite dans la définition des features : {leaked}")

    df = interim
    income = df["annual_inc"].where(df["annual_inc"] > 0)
    out = pd.DataFrame(
        {
            "id": df["id"],
            DATE_COL: df[DATE_COL],
            TARGET: df[TARGET].astype(int),
            "loan_amnt": df["loan_amnt"].astype(float),
            "installment": df["installment"].astype(float),
            "emp_length": df["emp_length"],
            "annual_inc_log": np.log1p(df["annual_inc"].clip(lower=0)),
            "dti": df["dti"],
            "delinq_2yrs": df["delinq_2yrs"],
            "credit_history_years": (df[DATE_COL] - df["earliest_cr_line"]).dt.days / 365.25,
            "fico_mean": (df["fico_range_low"] + df["fico_range_high"]) / 2.0,
            "inq_last_6mths": df["inq_last_6mths"],
            "open_acc": df["open_acc"],
            "pub_rec": df["pub_rec"],
            "revol_bal_log": np.log1p(df["revol_bal"].clip(lower=0)),
            "revol_util": df["revol_util"],
            "total_acc": df["total_acc"],
            "mort_acc": df["mort_acc"],
            "pub_rec_bankruptcies": df["pub_rec_bankruptcies"],
            "loan_to_income": df["loan_amnt"] / income,
            "installment_to_income": df["installment"] * 12.0 / income,
        }
    )
    for col in CATEGORICAL_FEATURES:
        out[col] = df[col].astype("string").fillna("unknown").astype(str)
    return out.reset_index(drop=True)
