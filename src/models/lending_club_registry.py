"""Enregistrement du champion Lending Club dans le registre MLflow (pyfunc + alias champion/previous).

``ScoringModel`` (prétraitement + estimateur + calibrateur) n'est pas un estimateur sklearn nu :
on l'enveloppe dans un modèle MLflow *pyfunc* générique, sérialisé par cloudpickle, exactement
comme n'importe quel modèle « fait maison » se déploie en production. La promotion réutilise
``models.promotion`` (mêmes portes de qualité, mêmes alias) : gouvernance unique pour les deux
pipelines (synthétique et Lending Club) plutôt qu'une logique dupliquée.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cloudpickle
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc.model import PythonModel, PythonModelContext

from features.lending_club import FEATURES
from models.promotion import PromotionGates, promote_if_better
from models.scoring import ScoringModel
from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "lending_club_champion"


class LendingClubPyfuncModel(PythonModel):
    """Enveloppe MLflow pyfunc autour d'un ``ScoringModel`` (prétraitement + calibration inclus)."""

    def load_context(self, context: PythonModelContext) -> None:
        with open(context.artifacts["scoring_model"], "rb") as fh:
            self.model: ScoringModel = cloudpickle.load(fh)

    def predict(
        self, context: PythonModelContext, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> Any:
        return self.model.predict_proba(model_input)


def register_champion(
    champion: ScoringModel,
    sample: pd.DataFrame,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    gates: PromotionGates | None = None,
) -> str:
    """Journalise le champion en pyfunc, l'enregistre, puis applique la porte de promotion.

    L'appelant doit avoir déjà journalisé, sur le run MLflow actif, les métriques ``roc_auc``
    et ``ece`` du champion (clés brutes, pas préfixées) : c'est ce que relit la porte de
    promotion (``models.promotion``) pour décider. Voir ``pipelines.benchmark.run_benchmark``.

    Args:
        champion: le ``ScoringModel`` gagnant du benchmark (déjà calibré).
        sample: quelques lignes de features (pour la signature MLflow, entrée/sortie typées).
        model_name: nom du modèle dans le registre (distinct de ``cif_credit_official``).
        gates: portes de qualité de promotion (défaut : ``PromotionGates()``).

    Returns:
        La version enregistrée (peut ne pas être promue championne si les portes refusent).
    """
    example = sample[list(FEATURES)].head(5)
    signature = infer_signature(example, champion.predict_proba(example))

    with tempfile.TemporaryDirectory() as tmp:
        pkl_path = Path(tmp) / "scoring_model.pkl"
        with pkl_path.open("wb") as fh:
            cloudpickle.dump(champion, fh)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=LendingClubPyfuncModel(),
            artifacts={"scoring_model": str(pkl_path)},
            signature=signature,
            input_example=example,
            registered_model_name=model_name,
        )
    version = str(info.registered_model_version)

    client = mlflow.tracking.MlflowClient()
    decision = promote_if_better(client, model_name, version, gates or PromotionGates())
    logger.info(
        "models.lending_club.register",
        version=version,
        promoted=decision.promoted,
        reason=decision.reason,
    )
    return version
