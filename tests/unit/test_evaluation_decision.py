"""Tests des métriques d'évaluation et de la décision."""

import numpy as np

from evaluation.bootstrap import bootstrap_metrics
from evaluation.metrics import (
    calibration_metrics,
    classification_at_threshold,
    compute_all_metrics,
)
from services.decision_engine import Decision, DecisionEngine, DecisionPolicy


def test_compute_all_metrics_ranges():
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1] * 10)
    y_prob = np.clip(np.linspace(0.05, 0.95, len(y_true)) + np.random.RandomState(0).normal(0, 0.1, len(y_true)), 0, 1)
    metrics = compute_all_metrics(y_true, y_prob)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["brier"] <= 0.5
    assert 0 <= metrics["ece"] <= 1


def test_calibration_metrics():
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1] * 10)
    y_prob = np.array([0.2, 0.8] * 50)
    ece, slope, _intercept = calibration_metrics(y_true, y_prob)
    assert 0 <= ece <= 1
    assert slope > 0


def test_classification_at_threshold():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.6])
    out = classification_at_threshold(y_true, y_prob, threshold=0.5)
    assert out["tp"] >= 0 and out["fnr"] >= 0
    assert 0 <= out["precision"] <= 1


def test_bootstrap_returns_ci():
    rng = np.random.RandomState(7)
    y_true = rng.randint(0, 2, 400)
    y_prob = y_true * 0.9 + (1 - y_true) * 0.1 + rng.normal(0, 0.05, 400)
    y_prob = np.clip(y_prob, 0, 1)
    result = bootstrap_metrics(y_true, y_prob, n_iterations=50, seed=7)
    assert "roc_auc" in result
    assert result["roc_auc"]["ci_low"] <= result["roc_auc"]["mean"] <= result["roc_auc"]["ci_high"]


def test_decision_engine():
    policy = DecisionPolicy(approve_threshold=0.1, review_threshold=0.25, hard_reject_threshold=0.5)
    engine = DecisionEngine(policy=policy)

    assert engine.decide(0.05, n_past_loans=3, customer_id=1).decision == Decision.APPROBATION
    assert engine.decide(0.15, n_past_loans=3, customer_id=1).decision == Decision.AJUSTEMENT
    assert engine.decide(0.30, n_past_loans=3, customer_id=1).decision == Decision.REVUE_HUMAINE
    assert engine.decide(0.60, n_past_loans=3, customer_id=1).decision == Decision.REFUS

    # Règle thin-file : même un bon score passe en revue humaine
    assert engine.decide(0.05, n_past_loans=0, customer_id=1).decision == Decision.REVUE_HUMAINE

    outcome = engine.decide(0.30, n_past_loans=3, customer_id=42)
    assert outcome.customer_id == 42
    assert outcome.score == 70
    assert outcome.policy_hit
