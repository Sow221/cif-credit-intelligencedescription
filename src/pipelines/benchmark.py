"""Benchmark de validation : baseline logistique vs XGBoost contraint, protocole out-of-time.

Protocole (fixé *avant* de regarder le test) :
1. Partition par date d'octroi : entraînement / validation / test (le plus récent).
2. Baseline logistique : coefficient de régularisation choisi sur la validation.
3. XGBoost : hyperparamètres choisis par CV temporelle *dans l'entraînement* (Optuna).
4. Calibration isotonique ajustée sur la validation, jamais sur l'entraînement ni le test.
5. Le test est évalué une seule fois. XGBoost n'est déclaré champion que si l'IC95% apparié de
   l'écart d'AUC sur le test exclut 0 ; sinon la baseline, plus simple et plus lisible, est retenue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from config.schema import LendingClubConfig
from data.lending_club import DATE_COL, TARGET, temporal_partition
from evaluation.bootstrap import bootstrap_metrics, paired_auc_difference
from evaluation.metrics import compute_all_metrics
from evaluation.thresholds import optimal_cost_threshold
from models.scoring import ScoringModel, fit_logistic, fit_xgboost
from models.tuning import tune_xgboost
from utils.logging import get_logger

logger = get_logger(__name__)

LOGISTIC_C_GRID = (0.01, 0.1, 1.0, 10.0)


@dataclass
class BenchmarkResult:
    """Sorties du benchmark (métriques sérialisables + modèles pour l'export)."""

    metrics: dict[str, Any]
    champion: ScoringModel
    challenger: ScoringModel
    test: pd.DataFrame
    test_scores: dict[str, np.ndarray]


def _split_summary(name: str, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "split": name,
        "rows": len(df),
        "default_rate": round(float(df[TARGET].mean()), 4),
        "from": str(df[DATE_COL].min().date()),
        "to": str(df[DATE_COL].max().date()),
    }


def _quarterly_auc(test: pd.DataFrame, score: np.ndarray) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {"q": test[DATE_COL].dt.to_period("Q").astype(str).to_numpy(), "y": test[TARGET].to_numpy(), "p": score}
    )
    rows = []
    for q, g in frame.groupby("q"):
        if g["y"].nunique() == 2:
            rows.append({"quarter": q, "n": len(g), "roc_auc": round(float(roc_auc_score(g["y"], g["p"])), 4)})
    return rows


def run_benchmark(
    features: pd.DataFrame,
    cfg: LendingClubConfig,
    *,
    n_trials: int | None = None,
    n_bootstrap: int = 200,
) -> BenchmarkResult:
    """Exécute le protocole complet et retourne métriques + modèles."""
    train, valid, test = temporal_partition(features, cfg)
    y_tr, y_va, y_te = (d[TARGET].to_numpy() for d in (train, valid, test))
    logger.info("benchmark.split", train=len(train), valid=len(valid), test=len(test))

    # --- Baseline logistique : C choisi sur la validation ---
    best_c, best_auc = LOGISTIC_C_GRID[0], -1.0
    for c in LOGISTIC_C_GRID:
        auc = float(roc_auc_score(y_va, fit_logistic(train, y_tr, c).raw_proba(valid)))
        if auc > best_auc:
            best_c, best_auc = c, auc
    logistic = fit_logistic(train, y_tr, best_c).calibrate(valid, y_va)

    # --- XGBoost : hyperparamètres par CV temporelle dans l'entraînement ---
    trials = n_trials if n_trials is not None else cfg.tuning_trials
    best_params, cv_auc = tune_xgboost(train, trials, cfg.tuning_max_rows, seed=cfg.seed)
    xgb = fit_xgboost(train, y_tr, best_params, seed=cfg.seed)
    xgb_raw_test = xgb.raw_proba(test)  # avant calibration, pour montrer son effet
    xgb.calibrate(valid, y_va)

    # --- Évaluation unique sur le test out-of-time ---
    scores = {"logistic": logistic.predict_proba(test), "xgboost": xgb.predict_proba(test)}
    model_metrics = {name: compute_all_metrics(y_te, p) for name, p in scores.items()}
    model_metrics["xgboost_uncalibrated"] = compute_all_metrics(y_te, np.clip(xgb_raw_test, 1e-4, 1 - 1e-4))
    ci = {name: bootstrap_metrics(y_te, p, n_iterations=n_bootstrap, seed=cfg.seed) for name, p in scores.items()}
    diff = paired_auc_difference(y_te, scores["xgboost"], scores["logistic"], n_iterations=n_bootstrap, seed=cfg.seed)

    xgb_wins = diff["ci_low"] > 0.0
    champion, challenger = (xgb, logistic) if xgb_wins else (logistic, xgb)
    logger.info("benchmark.champion", champion=champion.name, auc_diff=diff)

    # Seuil de décision par coût (Model ≠ Policy, cf. ADR CIF §64) : calculé une seule fois sur
    # le test, à partir du champion — jamais posé "à la main".
    champion_score = scores[champion.name]
    cost_threshold, expected_cost = optimal_cost_threshold(
        y_te, champion_score, cfg.cost_false_negative, cfg.cost_false_positive
    )

    metrics: dict[str, Any] = {
        "protocol": {
            "partition": [_split_summary("train", train), _split_summary("valid", valid), _split_summary("test", test)],
            "logistic_C": best_c,
            "xgboost_params": best_params,
            "xgboost_cv_auc_temporal": round(cv_auc, 4),
            "tuning_trials": trials,
            "exclude_lender_score": cfg.exclude_lender_score,
        },
        "test_metrics": model_metrics,
        "test_bootstrap_ci": ci,
        "paired_auc_diff_xgboost_minus_logistic": diff,
        "champion": champion.name,
        "champion_reason": (
            "IC95% apparié de l'écart d'AUC exclut 0 : XGBoost démontrablement meilleur"
            if xgb_wins
            else "Écart d'AUC non démontré (IC95% inclut 0) : baseline logistique retenue par parcimonie"
        ),
        "quarterly_test_auc": {name: _quarterly_auc(test, p) for name, p in scores.items()},
        "decision": {
            "cost_threshold": round(cost_threshold, 4),
            "expected_cost_per_case": round(expected_cost, 2),
            "cost_false_negative": cfg.cost_false_negative,
            "cost_false_positive": cfg.cost_false_positive,
            "note": "Coûts illustratifs (dollars arbitraires), pas une calibration métier réelle.",
        },
    }
    return BenchmarkResult(metrics, champion, challenger, test, scores)


def write_artifacts(result: BenchmarkResult, out_dir: str | Path) -> Path:
    """Écrit metrics.json et les figures (ROC, calibration, distribution du score) dans ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result.metrics, indent=2, default=float), encoding="utf-8")

    y = result.test[TARGET].to_numpy()
    # Bornes explicites (pas un simple entier) : évite un bug connu de np.histogram quand des
    # probabilités clippées s'empilent exactement sur le dernier bord de bin (numpy#10322).
    hist_edges = np.linspace(0.0, 1.0, 51)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, p in result.test_scores.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        bins = pd.qcut(p, 10, duplicates="drop")
        cal = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
        axes[1].plot(cal["p"], cal["y"], marker="o", label=name)
        axes[2].hist(p, bins=hist_edges, alpha=0.5, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="ROC (test out-of-time)", xlabel="FPR", ylabel="TPR")
    axes[1].plot([0, 0.6], [0, 0.6], "k--", lw=0.8)
    axes[1].set(title="Calibration (déciles)", xlabel="PD prédite", ylabel="Taux de défaut observé")
    axes[2].set(title="Distribution du score", xlabel="PD prédite")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "benchmark.png", dpi=130)
    plt.close(fig)
    return out
