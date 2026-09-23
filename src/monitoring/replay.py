"""Rejeu de monitoring sur historique (« backtest » de surveillance) — Lending Club.

Rejoue le protocole de surveillance de production sur les 24 mois qui suivent l'entraînement
du champion (2014-01 à 2015-12), en distinguant honnêtement les deux familles de signaux :

- **Dérive des données** (indicateur avancé, PSI — Population Stability Index,
  ``monitoring.drift.compute_psi_report``) : disponible immédiatement, dès qu'une cohorte de
  prêts est octroyée. Calculée sur les 24 mois complets (validation + test), car elle ne dépend
  d'aucune étiquette. Un rapport HTML Evidently complet est en plus généré pour le mois le plus
  dérivé (visualisation qualitative) ; la décision d'alerte repose uniquement sur le PSI, calculé
  indépendamment — un rapport HTML manquant ne remet jamais en cause l'alerte elle-même.
- **Performance réelle** (indicateur retardé) : calculée **seulement sur les 12 mois du test**
  (2015), jamais touchés à l'entraînement ni à la calibration — le seul sous-ensemble
  honnêtement hors-échantillon. En déploiement réel, chacun de ces 12 points n'aurait été
  disponible que 36 mois après l'octroi (durée du prêt) : nous ne pouvons les calculer
  aujourd'hui, tous ensemble, que parce que nous disposons du fichier historique complet
  (arrêté au T4 2018, garantissant la maturité — voir ADR 0001). Un vrai système de
  production aurait vu ces points arriver un par un, avec ce décalage.

Les seuils appliqués sont ceux déjà déclarés dans ``config.settings.MonitoringSettings``
(``psi_alert``, ``roc_auc_alert``, ``ece_alert``) — jamais réinventés pour l'occasion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config.settings import MonitoringSettings
from data.lending_club import DATE_COL, TARGET
from evaluation.metrics import compute_all_metrics
from features.lending_club import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from models.scoring import ScoringModel
from monitoring.drift import compute_data_drift_report, compute_psi_report
from utils.logging import get_logger

logger = get_logger(__name__)

DRIFT_FEATURE_COLS = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)


@dataclass
class ReplayResult:
    """Résultat du rejeu : deux séries mensuelles + verdicts d'alerte."""

    drift_series: pd.DataFrame  # colonnes : month, drift_ratio, alert
    performance_series: pd.DataFrame  # colonnes : month, roc_auc, ece, alert_roc_auc, alert_ece
    thresholds: dict[str, float]
    any_alert: bool
    representative_month: str  # mois le plus dérivé — celui dont le rapport HTML est généré


def _month_str(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M").astype(str)


def run_replay(
    features: pd.DataFrame,
    train_reference: pd.DataFrame,
    champion: ScoringModel,
    *,
    valid_start: str = "2014-01-01",
    valid_end: str = "2014-12-31",
    test_start: str = "2015-01-01",
    test_end: str = "2015-12-31",
    monitoring: MonitoringSettings | None = None,
) -> ReplayResult:
    """Rejoue la surveillance mois par mois sur la période validation+test (voir docstring module)."""
    monitoring = monitoring or MonitoringSettings()
    ref = train_reference[DRIFT_FEATURE_COLS].copy()

    monitored = features[
        (features[DATE_COL] >= pd.Timestamp(valid_start)) & (features[DATE_COL] <= pd.Timestamp(test_end))
    ].copy()
    monitored["month"] = _month_str(monitored[DATE_COL])

    drift_rows: list[dict[str, Any]] = []
    for month, group in monitored.groupby("month", sort=True):
        psi_by_col = compute_psi_report(ref, group[DRIFT_FEATURE_COLS], DRIFT_FEATURE_COLS)
        worst_col = max(psi_by_col, key=lambda c: psi_by_col[c])
        max_psi = psi_by_col[worst_col]
        drift_rows.append(
            {
                "month": month,
                "psi": max_psi,
                "worst_feature": worst_col,
                "alert": max_psi >= monitoring.psi_alert,
            }
        )
    drift_series = pd.DataFrame(drift_rows).sort_values("month").reset_index(drop=True)

    test = features[
        (features[DATE_COL] >= pd.Timestamp(test_start)) & (features[DATE_COL] <= pd.Timestamp(test_end))
    ].copy()
    test["month"] = _month_str(test[DATE_COL])

    perf_rows: list[dict[str, Any]] = []
    for month, group in test.groupby("month", sort=True):
        probs = champion.predict_proba(group)
        m = compute_all_metrics(group[TARGET].to_numpy(), probs)
        perf_rows.append(
            {
                "month": month,
                "n": len(group),
                "roc_auc": m["roc_auc"],
                "ece": m["ece"],
                "alert_roc_auc": m["roc_auc"] < monitoring.roc_auc_alert,
                "alert_ece": m["ece"] > monitoring.ece_alert,
            }
        )
    performance_series = pd.DataFrame(perf_rows).sort_values("month").reset_index(drop=True)

    any_alert = bool(
        drift_series["alert"].any()
        or performance_series["alert_roc_auc"].any()
        or performance_series["alert_ece"].any()
    )
    representative_month = str(drift_series.loc[drift_series["psi"].idxmax(), "month"])

    thresholds = {
        "psi_alert": monitoring.psi_alert,
        "roc_auc_alert": monitoring.roc_auc_alert,
        "ece_alert": monitoring.ece_alert,
    }
    logger.info(
        "monitoring.replay.done",
        any_alert=any_alert,
        n_months_drift=len(drift_series),
        n_months_perf=len(performance_series),
        representative_month=representative_month,
    )
    return ReplayResult(drift_series, performance_series, thresholds, any_alert, representative_month)


def write_replay_artifacts(
    result: ReplayResult,
    features: pd.DataFrame,
    train_reference: pd.DataFrame,
    out_dir: str | Path,
) -> Path:
    """Écrit metrics.json, un graphique à deux panneaux, et un rapport Evidently HTML complet
    pour le mois le plus dérivé (utiliser la vraie plateforme de rapport, pas juste un chiffre)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "thresholds": result.thresholds,
        "any_alert": result.any_alert,
        "representative_month": result.representative_month,
        "drift_series": result.drift_series.to_dict(orient="records"),
        "performance_series": result.performance_series.to_dict(orient="records"),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    ax1.plot(result.drift_series["month"], result.drift_series["psi"], marker="o", color="tab:blue")
    ax1.axhline(result.thresholds["psi_alert"], color="red", ls="--", lw=1, label="seuil d'alerte (PSI)")
    ax1.set_title("Indicateur avancé — PSI maximal par mois (24 mois, disponible immédiatement)")
    ax1.set_ylabel("PSI (pire variable du mois)")
    ax1.tick_params(axis="x", rotation=60)
    ax1.legend()

    ax2b = ax2.twinx()
    ax2.plot(
        result.performance_series["month"],
        result.performance_series["roc_auc"],
        marker="o",
        color="tab:green",
        label="ROC-AUC",
    )
    ax2.axhline(result.thresholds["roc_auc_alert"], color="red", ls="--", lw=1)
    ax2b.plot(
        result.performance_series["month"],
        result.performance_series["ece"],
        marker="s",
        color="tab:orange",
        label="ECE",
    )
    ax2b.axhline(result.thresholds["ece_alert"], color="darkred", ls=":", lw=1)
    ax2.set_title("Indicateur retardé — performance réelle (12 mois test, connue avec 36 mois de retard en production)")
    ax2.set_ylabel("ROC-AUC", color="tab:green")
    ax2b.set_ylabel("ECE", color="tab:orange")
    ax2.tick_params(axis="x", rotation=60)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="center left")

    fig.tight_layout()
    fig.savefig(out / "replay.png", dpi=130)
    plt.close(fig)

    # Rapport HTML Evidently (visualisation qualitative complète) pour le mois le plus dérivé —
    # séparé du calcul de PSI ci-dessus (qui fait foi pour la décision d'alerte) : un incident de
    # rendu ne doit jamais invalider la mesure elle-même. Vu en pratique : le calcul interne
    # d'Evidently (np.histogram) échoue sur des variables à valeurs très regroupées sur des
    # paliers ronds (ex. plafonds de prêt à 10 000 $/15 000 $/35 000 $) — bug numpy connu
    # (numpy#10322), hors de notre contrôle, jamais déclenché par notre propre code de mesure.
    rep_month = result.representative_month
    current = features[features[DATE_COL].dt.to_period("M").astype(str) == rep_month]
    try:
        compute_data_drift_report(
            train_reference[DRIFT_FEATURE_COLS],
            current[DRIFT_FEATURE_COLS],
            str(out / f"drift_report_{rep_month}"),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("monitoring.replay.html_report_failed", month=rep_month, error=str(exc))
    return out
