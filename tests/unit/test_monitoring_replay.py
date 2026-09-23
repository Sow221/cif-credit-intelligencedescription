"""Tests du PSI (monitoring.drift) et du rejeu de monitoring (monitoring.replay)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import compute_psi, compute_psi_report
from monitoring.replay import run_replay


def test_psi_is_near_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(size=5000))
    cur = pd.Series(rng.normal(size=5000))
    assert compute_psi(ref, cur) < 0.05


def test_psi_is_high_for_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(loc=0, size=5000))
    cur = pd.Series(rng.normal(loc=5, size=5000))  # décalage massif, sans ambiguïté
    assert compute_psi(ref, cur) > 0.25


def test_psi_handles_categorical_columns() -> None:
    ref = pd.Series(["a"] * 500 + ["b"] * 500)
    cur_same = pd.Series(["a"] * 500 + ["b"] * 500)
    cur_shifted = pd.Series(["a"] * 950 + ["b"] * 50)
    assert compute_psi(ref, cur_same) < 0.01
    assert compute_psi(ref, cur_shifted) > 0.25


def test_psi_handles_values_clustered_at_round_caps() -> None:
    """Cas réel qui fait planter Evidently (numpy#10322) : beaucoup de valeurs identiques à un
    palier rond. compute_psi doit rester robuste, contrairement au calcul interne d'Evidently."""
    rng = np.random.default_rng(0)
    ref = pd.Series(np.concatenate([rng.uniform(1000, 34000, 800), np.full(200, 35000.0)]))
    cur = pd.Series(np.concatenate([rng.uniform(1000, 34000, 800), np.full(200, 35000.0)]))
    assert compute_psi(ref, cur) < 0.1  # ne plante pas, résultat cohérent


def test_psi_report_returns_one_value_per_column() -> None:
    ref = pd.DataFrame({"x": np.random.default_rng(0).normal(size=200), "cat": ["a"] * 200})
    cur = pd.DataFrame({"x": np.random.default_rng(1).normal(size=200), "cat": ["a"] * 200})
    report = compute_psi_report(ref, cur, ["x", "cat"])
    assert set(report) == {"x", "cat"}
    assert all(v >= 0.0 for v in report.values())


class _StubChampion:
    """Modèle jouet : probabilité = fonction déterministe d'une seule variable, pour un test rapide."""

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(df["risk_score"].to_numpy(), 1e-4, 1 - 1e-4)


def _make_replay_features(n_per_month: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for year, n_months in ((2013, 12), (2014, 12), (2015, 12)):
        for m in range(1, n_months + 1):
            risk = rng.uniform(0, 1, n_per_month)
            y = (rng.random(n_per_month) < risk).astype(int)
            rows.append(
                pd.DataFrame(
                    {
                        "issue_d": pd.Timestamp(year=year, month=m, day=1),
                        "risk_score": risk,
                        "purpose": rng.choice(["a", "b", "c"], n_per_month),
                        "is_default": y,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_run_replay_produces_24_drift_months_and_12_performance_months(monkeypatch) -> None:
    monkeypatch.setattr("monitoring.replay.DRIFT_FEATURE_COLS", ["risk_score", "purpose"])
    features = _make_replay_features()
    train = features[features["issue_d"].dt.year == 2013]
    result = run_replay(features, train, _StubChampion())

    assert len(result.drift_series) == 24  # 2014 (validation) + 2015 (test)
    assert len(result.performance_series) == 12  # test seulement, jamais validation/entraînement
    assert set(result.performance_series["month"]) == {f"2015-{m:02d}" for m in range(1, 13)}
    assert result.thresholds["psi_alert"] == pytest.approx(0.25)


def test_run_replay_flags_alert_only_above_threshold(monkeypatch) -> None:
    monkeypatch.setattr("monitoring.replay.DRIFT_FEATURE_COLS", ["risk_score", "purpose"])
    features = _make_replay_features()
    train = features[features["issue_d"].dt.year == 2013]
    result = run_replay(features, train, _StubChampion())

    for _, row in result.drift_series.iterrows():
        assert row["alert"] == (row["psi"] >= result.thresholds["psi_alert"])
    for _, row in result.performance_series.iterrows():
        assert row["alert_roc_auc"] == (row["roc_auc"] < result.thresholds["roc_auc_alert"])
        assert row["alert_ece"] == (row["ece"] > result.thresholds["ece_alert"])


def test_run_replay_never_lets_calibration_period_leak_into_performance(monkeypatch) -> None:
    """2014 sert à la calibration : sa performance ne doit jamais être présentée comme
    hors-échantillon — seul 2015 (test, jamais touché) doit apparaître dans performance_series."""
    monkeypatch.setattr("monitoring.replay.DRIFT_FEATURE_COLS", ["risk_score", "purpose"])
    features = _make_replay_features()
    train = features[features["issue_d"].dt.year == 2013]
    result = run_replay(features, train, _StubChampion())

    assert not any(m.startswith("2014") for m in result.performance_series["month"])
    assert any(m.startswith("2014") for m in result.drift_series["month"])
