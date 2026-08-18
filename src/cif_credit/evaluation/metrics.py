"""Métriques d'évaluation alignées sur le protocole CIF (ROC/PR/Brier/LogLoss/Calibration)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Calcule le jeu complet de métriques du protocole de validation CIF.

    Args:
        y_true: étiquettes binaires réelles.
        y_prob: probabilités prédites.

    Returns:
        Dictionnaire des métriques.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))

    ece, slope, intercept = calibration_metrics(y_true, y_prob)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "log_loss": ll,
        "ece": ece,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, float, float]:
    """ECE (Expected Calibration Error), slope et intercept de calibration.

    Calcul manuel par bins uniformes (robuste même quand les probabilités prédites
    prennent peu de valeurs distinctes, cas fréquent en production).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    n = len(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)

    ece = 0.0
    prob_true_list: list[float] = []
    prob_pred_list: list[float] = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        prob_pred = float(y_prob[mask].mean())
        prob_true = float(y_true[mask].mean())
        weight = count / max(n, 1)
        ece += weight * abs(prob_true - prob_pred)
        prob_true_list.append(prob_true)
        prob_pred_list.append(prob_pred)

    if len(prob_pred_list) > 1:
        slope, intercept = np.polyfit(prob_pred_list, prob_true_list, 1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        slope, intercept = 1.0, 0.0
    return float(ece), slope, intercept


def classification_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Matrice de confusion dérivée et métriques seuillées (precision/recall/F1/MCC/FPR/FNR)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    mcc = (tp * tn - fp * fn) / max(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 1e-9)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    return {
        "threshold": float(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": float(mcc),
        "fpr": fpr,
        "fnr": fnr,
    }
