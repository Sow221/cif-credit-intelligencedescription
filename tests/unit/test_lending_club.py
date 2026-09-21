from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.schema import LendingClubConfig
from data.lending_club import (
    DATE_COL,
    POST_ORIGINATION_COLUMNS,
    TARGET,
    LendingClubDataError,
    clean,
    find_raw_file,
    load_raw,
    temporal_partition,
)
from data.validation import validate_clean
from evaluation.bootstrap import paired_auc_difference
from evaluation.metrics import compute_all_metrics, ks_statistic
from evaluation.thresholds import optimal_cost_threshold
from features.lending_club import FEATURES, MONOTONE_CONSTRAINTS, build_features
from models.scoring import fit_logistic, fit_xgboost
from pipelines.benchmark import run_benchmark, write_artifacts

CFG = LendingClubConfig()


@pytest.fixture(scope="module")
def interim(raw_lc: pd.DataFrame) -> pd.DataFrame:
    return validate_clean(clean(raw_lc, CFG), CFG)


@pytest.fixture(scope="module")
def features(interim: pd.DataFrame) -> pd.DataFrame:
    return build_features(interim)


def test_clean_keeps_only_mature_36m_window(raw_lc: pd.DataFrame, interim: pd.DataFrame) -> None:
    assert set(interim[TARGET].unique()) == {0, 1}
    assert (interim["term"] == 36).all()
    assert interim[DATE_COL].between(pd.Timestamp(CFG.issue_start), pd.Timestamp(CFG.issue_end)).all()
    assert len(interim) < len(raw_lc)  # 60 mois, "Current", "Late" et dates hors périmètre exclus


def test_clean_drops_lender_score_and_post_origination(interim: pd.DataFrame) -> None:
    assert not {"grade", "int_rate"} & set(interim.columns)
    assert not POST_ORIGINATION_COLUMNS & set(interim.columns)


def test_clean_parses_dirty_formats(interim: pd.DataFrame) -> None:
    assert interim["emp_length"].dropna().between(0, 10).all()
    assert pd.api.types.is_datetime64_any_dtype(interim["earliest_cr_line"])
    assert interim["id"].is_unique


def test_validation_rejects_post_origination_leak(interim: pd.DataFrame) -> None:
    leaky = interim.assign(total_pymnt=1.0)
    with pytest.raises(LendingClubDataError, match="après-octroi"):
        validate_clean(leaky, CFG)


def test_validation_rejects_out_of_range_values(interim: pd.DataFrame) -> None:
    bad = interim.copy()
    bad.loc[bad.index[0], "fico_range_low"] = 12.0
    with pytest.raises(LendingClubDataError):
        validate_clean(bad, CFG)


def test_load_raw_requires_mandatory_columns(tmp_path) -> None:
    pd.DataFrame({"id": [1], "loan_amnt": [1.0]}).to_csv(tmp_path / "accepted_x.csv", index=False)
    with pytest.raises(LendingClubDataError, match="obligatoires"):
        load_raw(tmp_path / "accepted_x.csv", CFG)


def test_find_raw_file_missing_gives_actionable_error(tmp_path) -> None:
    with pytest.raises(LendingClubDataError, match="data-download"):
        find_raw_file(tmp_path)


def test_features_are_point_in_time_and_leak_free(interim: pd.DataFrame, features: pd.DataFrame) -> None:
    assert not set(FEATURES) & POST_ORIGINATION_COLUMNS
    row = interim.iloc[0]
    expected = (row[DATE_COL] - row["earliest_cr_line"]).days / 365.25
    assert features.loc[0, "credit_history_years"] == pytest.approx(expected)
    assert features["loan_to_income"].dropna().gt(0).all()


def test_temporal_partition_is_ordered_and_disjoint(features: pd.DataFrame) -> None:
    tr, va, te = temporal_partition(features, CFG)
    assert tr[DATE_COL].max() < va[DATE_COL].min() <= va[DATE_COL].max() < te[DATE_COL].min()
    assert len(tr) + len(va) + len(te) == len(features)


def test_metrics_include_ks_and_gini() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    m = compute_all_metrics(y, p)
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)
    assert ks_statistic(y, p) == pytest.approx(1.0)


def test_cost_threshold_prefers_refusing_when_defaults_are_costly() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.random(5000) < p).astype(int)
    t_costly, _ = optimal_cost_threshold(y, p, cost_false_negative=10.0, cost_false_positive=1.0)
    t_cheap, _ = optimal_cost_threshold(y, p, cost_false_negative=1.0, cost_false_positive=1.0)
    assert t_costly < t_cheap


def test_paired_auc_difference_detects_better_model() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 3000)
    good = y + rng.normal(0, 0.6, 3000)
    bad = rng.normal(0, 1, 3000)
    d = paired_auc_difference(y, good, bad, n_iterations=100)
    assert d["ci_low"] > 0


def test_xgboost_respects_monotone_constraints(features: pd.DataFrame) -> None:
    tr, _, _ = temporal_partition(features, CFG)
    model = fit_xgboost(tr, tr[TARGET].to_numpy(), {"max_depth": 3, "n_estimators": 60, "learning_rate": 0.1})
    sample = tr.sample(300, random_state=0)
    for feat, sign in MONOTONE_CONSTRAINTS.items():
        lo, hi = sample.copy(), sample.copy()
        lo[feat] = sample[feat].quantile(0.1)
        hi[feat] = sample[feat].quantile(0.9)
        delta = model.raw_proba(hi) - model.raw_proba(lo)
        assert (sign * delta >= -1e-9).all(), f"contrainte violée pour {feat}"


def test_explanations_sum_to_model_margin(features: pd.DataFrame) -> None:
    tr, _, _ = temporal_partition(features, CFG)
    model = fit_xgboost(tr, tr[TARGET].to_numpy(), {"max_depth": 3, "n_estimators": 40, "learning_rate": 0.1})
    sample = tr.head(50)
    contrib = model.explain(sample)
    margin = contrib.sum(axis=1).to_numpy()
    assert 1 / (1 + np.exp(-margin)) == pytest.approx(model.raw_proba(sample), abs=1e-4)
    assert set(FEATURES) <= set(contrib.columns)


def test_logistic_calibration_is_fit_on_separate_set(features: pd.DataFrame) -> None:
    tr, va, _ = temporal_partition(features, CFG)
    model = fit_logistic(tr, tr[TARGET].to_numpy()).calibrate(va, va[TARGET].to_numpy())
    p = model.predict_proba(va)
    assert p.min() > 0 and p.max() < 1


def test_benchmark_end_to_end(features: pd.DataFrame, tmp_path) -> None:
    result = run_benchmark(features, CFG, n_trials=2, n_bootstrap=20)
    m = result.metrics
    assert m["champion"] in {"logistic", "xgboost"}
    assert 0.5 < m["test_metrics"]["logistic"]["roc_auc"] <= 1.0
    parts = {p["split"]: p for p in m["protocol"]["partition"]}
    assert parts["train"]["to"] < parts["valid"]["from"] <= parts["valid"]["to"] < parts["test"]["from"]
    out = write_artifacts(result, tmp_path)
    assert (out / "metrics.json").exists() and (out / "benchmark.png").exists()
