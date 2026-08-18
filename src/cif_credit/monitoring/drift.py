"""Monitoring de dérive de données et de qualité — Evidently.

Génère un rapport de drift (features + target) entre une référence (entraînement)
et une distribution courante (production). Alimente l'alerte Prometheus/Grafana.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cif_credit.utils.logging import get_logger

logger = get_logger(__name__)


def compute_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_path: str,
    *,
    target: str | None = None,
) -> dict[str, Any]:
    """Calcule le rapport de drift Evidently (DataDriftPreset + DataQualityPreset).

    Args:
        reference: données de référence (ex : train).
        current: données courantes (ex : batch de production).
        report_path: chemin de sortie HTML/JSON.
        target: nom de la colonne cible si elle doit être surveillée.

    Returns:
        Résumé des drift (nombre de features en drift, taux de drift).
    """
    try:
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report
    except ImportError as exc:  # pragma: no cover - dépendance optionnelle
        raise RuntimeError("evidently est requis pour le monitoring : pip install evidently") from exc

    reference_df = reference.copy()
    current_df = current.copy()
    if target:
        if target in reference_df.columns:
            reference_df = reference_df.rename(columns={target: "target"})
        if target in current_df.columns:
            current_df = current_df.rename(columns={target: "target"})
        # La target réelle peut être inconnue en production → colonne manquante tolérée
        if "target" not in current_df.columns:
            current_df["target"] = 0
        metrics: Any = [DataDriftPreset(), DataQualityPreset()]
    else:
        metrics = [DataDriftPreset(), DataQualityPreset()]

    report = Report(metrics=metrics)
    report.run(reference_data=reference_df, current_data=current_df)

    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out.with_suffix(".html")))

    summary: dict[str, Any] = {}
    raw = report.as_dict()
    try:
        drift_payload = raw["metrics"][0]["result"]
        summary["n_features"] = len(drift_payload.get("drift_by_columns", {}))
        n_drift = int(
            sum(
                1
                for v in drift_payload.get("drift_by_columns", {}).values()
                if v.get("drift_detected")
            )
        )
        summary["n_features_in_drift"] = n_drift
        summary["drift_ratio"] = round(
            n_drift / max(summary["n_features"], 1), 4
        )
        summary["dataset_drift"] = bool(drift_payload.get("dataset_drift"))
    except (KeyError, TypeError):
        summary = {"status": "report_generated", "detail": str(raw.get("errors", ""))}

    with out.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    logger.info("monitoring.drift.done", summary=summary, path=str(out))
    return summary


def check_drift_alert(summary: dict[str, Any], threshold: float = 0.2) -> bool:
    """True si le taux de drift dépasse le seuil d'alerte."""
    ratio = summary.get("drift_ratio", 0.0)
    return bool(ratio is not None and float(ratio) >= threshold)
