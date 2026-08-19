"""Rapport de monitoring complet — Evidently + PSI + alertes (Semaine 4).

Fichier exigé par le cabinet (retour.txt) : ``src/monitoring/drift_report.py``. Produit le
rapport Evidently de dérive de données et la surveillance de dérive de population (PSI),
puis évalue les alertes aux seuils du cabinet :
ROC-AUC < 0.65, ECE > 0.10, PSI > 0.25, taux de features en drift >= 0.20.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from monitoring.drift import check_drift_alert, compute_data_drift_report
from utils.logging import get_logger

logger = get_logger(__name__)

ALERT_THRESHOLDS: dict[str, float] = {
    "roc_auc_min": 0.65,
    "ece_max": 0.10,
    "psi_max": 0.25,
    "drift_ratio_max": 0.20,
}


def psi_score(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index — mesure la dérive de distribution d'une variable.

    PSI = sum((act_ratio - ref_ratio) * log(act_ratio / ref_ratio)).
    PSI > 0.25 = derive significative (alerte cabinet), 0.10 a 0.25 = moderee.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    bins = np.linspace(min(expected.min(), actual.min()), max(expected.max(), actual.max()), n_bins + 1)
    if bins[0] == bins[-1]:
        return 0.0
    expected_in_bins, _ = np.histogram(expected, bins=bins)
    actual_in_bins, _ = np.histogram(actual, bins=bins)

    ref_ratio = expected_in_bins / max(len(expected), 1)
    act_ratio = actual_in_bins / max(len(actual), 1)
    ref_ratio = np.clip(ref_ratio, 1e-6, None)
    act_ratio = np.clip(act_ratio, 1e-6, None)

    psi = float(np.sum((act_ratio - ref_ratio) * np.log(act_ratio / ref_ratio)))
    return round(psi, 6)


def check_alert(threshold_name: str, value: float, thresholds: dict[str, float] | None = None) -> bool:
    """Évalue une alerte selon la direction du seuil (min : sous = alerte ; max : au-dessus = alerte)."""
    thr = thresholds or ALERT_THRESHOLDS
    value = float(value)
    if threshold_name in ("roc_auc_min",):
        return value < thr[threshold_name]
    return value > thr[threshold_name]


def check_monitoring_alerts(
    metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Confronte les métriques (roc_auc, ece, psi, drift_ratio) aux seuils du cabinet.

    Returns:
        Dict {métrique: {value, threshold, alerted}} — une alerte True déclenche le reporting.
    """
    thr = thresholds or ALERT_THRESHOLDS
    alerts: dict[str, dict[str, Any]] = {}
    mappings = {
        "roc_auc": "roc_auc_min",
        "ece": "ece_max",
        "psi": "psi_max",
        "drift_ratio": "drift_ratio_max",
    }
    for metric, threshold_name in mappings.items():
        if metric not in metrics:
            continue
        value = float(metrics[metric])
        alerts[metric] = {
            "value": round(value, 6),
            "threshold": thr[threshold_name],
            "alerted": check_alert(threshold_name, value, thr),
        }
    return alerts


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_path: str | Path,
    *,
    target: str | None = None,
) -> dict[str, Any]:
    """Génère le rapport Evidently + résumé JSON (fichiers html/json côté du chemin)."""
    summary = compute_data_drift_report(reference, current, str(report_path), target=target)
    summary["alerted"] = check_drift_alert(summary, ALERT_THRESHOLDS["drift_ratio_max"])
    return summary


def generate_psi_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str | Path,
    *,
    columns: list[str] | None = None,
) -> dict[str, float]:
    """Calcule le PSI de chaque colonne partagée et l'écrit en JSON."""
    cols = columns or sorted(set(reference.columns) & set(current.columns))
    psi_values: dict[str, float] = {}
    for col in cols:
        psi_values[col] = psi_score(reference[col].to_numpy(dtype=float), current[col].to_numpy(dtype=float))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(psi_values, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("monitoring.psi.done", n_columns=len(psi_values), path=str(out))
    return psi_values


def full_monitoring_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    base_path: str | Path,
    *,
    metrics: dict[str, float],
    target: str | None = None,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble drift Evidently + PSI + alertes, persiste l'ensemble en JSON unique.

    Returns:
        Dict complet {drift, psi, alerts, thresholds}.
    """
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)

    drift = generate_drift_report(reference, current, base / "drift", target=target)
    psi = generate_psi_report(reference, current, base / "psi.json", columns=columns)

    metrics_with_psi = {**metrics, "psi": max(psi.values()) if psi else 0.0, **drift}
    alerts = check_monitoring_alerts(metrics_with_psi)

    report = {"drift": drift, "psi": psi, "alerts": alerts, "thresholds": ALERT_THRESHOLDS}
    (base / "monitoring_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
