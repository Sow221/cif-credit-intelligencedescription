"""Promotion de modèle par porte de qualité, via les *alias* du registre MLflow.

Remplace les « stages » (dépréciés par MLflow) : l'alias ``champion`` désigne la version servie,
``previous`` la version précédente (rollback en une opération). Une version candidate ne devient
champion que si elle passe les portes ci-dessous ; chaque décision est tracée en tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlflow.tracking import MlflowClient

from utils.logging import get_logger

logger = get_logger(__name__)

CHAMPION = "champion"
PREVIOUS = "previous"


@dataclass(frozen=True)
class PromotionGates:
    """Portes de qualité : le candidat doit battre le champion ET rester calibré."""

    metric: str = "roc_auc"
    min_gain: float = 0.0
    max_ece: float = 0.10
    min_metric: float = 0.60


@dataclass
class PromotionDecision:
    promoted: bool
    version: str
    reason: str


def _run_metrics(client: MlflowClient, model_name: str, version: str) -> dict[str, float]:
    mv = client.get_model_version(model_name, version)
    if not mv.run_id:
        return {}
    return dict(client.get_run(mv.run_id).data.metrics)


def _metric(metrics: dict[str, float], name: str) -> float | None:
    for key in (name, f"test_{name}"):
        if key in metrics:
            return float(metrics[key])
    return None


def latest_version(client: MlflowClient, model_name: str) -> str:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"Aucune version enregistrée pour {model_name}")
    return str(max(int(v.version) for v in versions))


def current_champion(client: MlflowClient, model_name: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(model_name, CHAMPION).version)
    except Exception:  # alias absent
        return None


def promote_if_better(
    client: MlflowClient,
    model_name: str,
    candidate_version: str,
    gates: PromotionGates | None = None,
) -> PromotionDecision:
    """Évalue les portes puis, si elles passent, pose l'alias ``champion`` (l'ancien devient ``previous``)."""
    gates = gates or PromotionGates()
    cand = _run_metrics(client, model_name, candidate_version)
    cand_score = _metric(cand, gates.metric)
    cand_ece = _metric(cand, "ece")

    def refuse(reason: str) -> PromotionDecision:
        client.set_model_version_tag(model_name, candidate_version, "promotion", f"refused: {reason}")
        logger.info("models.promotion.refused", version=candidate_version, reason=reason)
        return PromotionDecision(False, candidate_version, reason)

    if cand_score is None:
        return refuse(f"métrique {gates.metric} absente du run")
    if cand_score < gates.min_metric:
        return refuse(f"{gates.metric}={cand_score:.4f} < plancher {gates.min_metric}")
    if cand_ece is not None and cand_ece > gates.max_ece:
        return refuse(f"ece={cand_ece:.4f} > {gates.max_ece}")

    champion = current_champion(client, model_name)
    if champion == candidate_version:
        return PromotionDecision(True, candidate_version, "déjà champion")
    if champion is not None:
        champ_score = _metric(_run_metrics(client, model_name, champion), gates.metric)
        if champ_score is not None and cand_score < champ_score + gates.min_gain:
            return refuse(f"{gates.metric}={cand_score:.4f} ne bat pas le champion v{champion} ({champ_score:.4f})")
        client.set_registered_model_alias(model_name, PREVIOUS, champion)

    client.set_registered_model_alias(model_name, CHAMPION, candidate_version)
    client.set_model_version_tag(model_name, candidate_version, "promotion", "champion")
    logger.info("models.promotion.promoted", version=candidate_version, previous=champion)
    return PromotionDecision(True, candidate_version, "portes franchies")


def rollback(client: MlflowClient, model_name: str) -> str:
    """Rétablit la version ``previous`` comme champion ; retourne la version restaurée."""
    try:
        prev = str(client.get_model_version_by_alias(model_name, PREVIOUS).version)
    except Exception as exc:
        raise ValueError("Aucune version 'previous' : rollback impossible") from exc
    current = current_champion(client, model_name)
    client.set_registered_model_alias(model_name, CHAMPION, prev)
    if current is not None:
        client.set_model_version_tag(model_name, current, "promotion", "rolled_back")
    logger.info("models.promotion.rollback", restored=prev, replaced=current)
    return prev


def gates_from_dict(values: dict[str, Any]) -> PromotionGates:
    return PromotionGates(**{k: v for k, v in values.items() if k in PromotionGates.__dataclass_fields__})
