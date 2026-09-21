"""Jeu Lending Club *fictif* au format du fichier réel — uniquement pour tester le code.

Il reproduit les formats sales du fichier Kaggle (``" 36 months"``, ``"10+ years"``, ``"Dec-2013"``,
statuts variés, colonnes d'après-octroi) avec un signal de risque cohérent. Ce n'est pas un jeu
de données de démonstration : aucun résultat scientifique ne doit en être tiré.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_raw_lending_club(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2009-01-01", "2016-06-01", freq="MS")
    issue = rng.choice(months, n)
    fico = rng.normal(700, 35, n).clip(620, 830).round(-1)
    dti = rng.gamma(4, 5, n).clip(0, 60)
    income = np.exp(rng.normal(11, 0.5, n))
    amount = rng.choice([5000, 8000, 12000, 20000, 30000], n).astype(float)
    util = rng.uniform(0, 100, n)
    risk = -0.02 * (fico - 700) + 0.03 * dti + 0.008 * util + 1.2 * (amount / income) - 2.3
    p = 1 / (1 + np.exp(-risk))
    bad = rng.random(n) < p
    status = np.where(bad, "Charged Off", "Fully Paid").astype(object)
    status[rng.random(n) < 0.05] = "Current"
    status[rng.random(n) < 0.02] = "Late (31-120 days)"
    term = np.where(rng.random(n) < 0.75, " 36 months", " 60 months")
    emp = rng.choice(["< 1 year", "1 year", "5 years", "10+ years", None], n)
    fmt = lambda s: pd.to_datetime(s).strftime("%b-%Y")  # noqa: E731
    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1).astype(str),
            "loan_amnt": amount,
            "term": term,
            "installment": (amount / 36).round(2),
            "emp_length": emp,
            "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
            "annual_inc": income.round(0),
            "verification_status": rng.choice(["Verified", "Not Verified"], n),
            "issue_d": fmt(issue),
            "loan_status": status,
            "purpose": rng.choice(["debt_consolidation", "credit_card", "car", "other"], n),
            "dti": dti.round(2),
            "delinq_2yrs": rng.poisson(0.3, n).astype(float),
            "earliest_cr_line": fmt(pd.to_datetime(issue) - pd.to_timedelta(rng.integers(700, 9000, n), unit="D")),
            "fico_range_low": fico,
            "fico_range_high": fico + 4,
            "inq_last_6mths": rng.poisson(0.8, n).astype(float),
            "open_acc": rng.integers(2, 25, n).astype(float),
            "pub_rec": rng.poisson(0.1, n).astype(float),
            "revol_bal": rng.integers(0, 40000, n).astype(float),
            "revol_util": util.round(1),
            "total_acc": rng.integers(5, 50, n).astype(float),
            "application_type": "Individual",
            "mort_acc": rng.poisson(1, n).astype(float),
            "pub_rec_bankruptcies": rng.poisson(0.05, n).astype(float),
            "grade": rng.choice(list("ABCDE"), n),
            "int_rate": rng.uniform(6, 25, n).round(2),
            # Colonnes d'APRÈS-octroi présentes dans le fichier réel : ne doivent jamais passer.
            "total_pymnt": rng.uniform(0, 30000, n).round(2),
            "recoveries": rng.uniform(0, 500, n).round(2),
        }
    )


@pytest.fixture(scope="session")
def raw_lc() -> pd.DataFrame:
    return make_raw_lending_club()
