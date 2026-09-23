"""Monitoring de dérive de données et de qualité — Evidently.

Génère un rapport de drift (features + target) entre une référence (entraînement)
et une distribution courante (production). Alimente l'alerte Prometheus/Grafana.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.logging import get_logger

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
        summary["n_features"] = int(drift_payload.get("number_of_columns", 0))
        n_drift = int(drift_payload.get("number_of_drifted_columns", 0))
        summary["n_features_in_drift"] = n_drift
        summary["drift_ratio"] = round(n_drift / max(summary["n_features"], 1), 4)
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


def compute_drift_ratio(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    """Calcule uniquement le taux de features en drift (DataDriftPreset), sans écriture de fichier.

    Utilisé par le monitoring temps réel pour alimenter la jauge Prometheus ``cif_drift_ratio``.
    """
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference.copy(), current_data=current.copy())
    raw = report.as_dict()
    try:
        payload = raw["metrics"][0]["result"]
        n = int(payload.get("number_of_columns", 0))
        n_drift = int(payload.get("number_of_drifted_columns", 0))
        return round(n_drift / max(n, 1), 4)
    except (KeyError, TypeError):
        return 0.0


def _psi_from_shares(ref_pct: pd.Series, cur_pct: pd.Series) -> float:
    """Formule PSI standard : Σ (cur% - ref%) · ln(cur% / ref%), sur des parts déjà non nulles."""
    import numpy as np

    eps = 1e-4  # évite ln(0) sans fausser significativement une part réelle
    ref = ref_pct.clip(lower=eps)
    cur = cur_pct.clip(lower=eps)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI (Population Stability Index) d'une variable — calcul direct, sans Evidently.

    Standard du secteur, indépendant de tout choix de test statistique par colonne : découpe
    la référence en ``bins`` classes de fréquence égale (déciles par défaut), compare la part de
    la population courante dans chaque classe. Repères usuels : PSI < 0.1 pas de dérive
    notable, 0.1-0.25 dérive modérée à surveiller, > 0.25 dérive significative.

    Fonctionne pour le numérique (classes par quantile) et le catégoriel (classes = catégories) ;
    détecté automatiquement par le type de la série.
    """
    import numpy as np

    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        return 0.0

    if pd.api.types.is_numeric_dtype(ref):
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:  # variable quasi constante : pas de découpage informatif possible
            return 0.0
        edges = edges.copy()
        edges[0], edges[-1] = -np.inf, np.inf
        ref_bins = pd.cut(ref, edges, duplicates="drop")
        cur_bins = pd.cut(cur, edges, duplicates="drop")
    else:
        ref_bins = ref.astype(str)
        cur_bins = cur.astype(str)

    ref_pct = ref_bins.value_counts(normalize=True, sort=False)
    cur_pct = cur_bins.value_counts(normalize=True, sort=False).reindex(ref_pct.index, fill_value=0.0)
    return round(_psi_from_shares(ref_pct, cur_pct), 4)


def compute_psi_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """PSI par colonne. La colonne au PSI le plus élevé domine la décision d'alerte (convention
    standard : une seule variable très dérivée doit alerter, une moyenne la diluerait)."""
    return {col: compute_psi(reference[col], current[col]) for col in columns if col in reference.columns}
