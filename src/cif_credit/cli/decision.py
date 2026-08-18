"""CLI : cif-decision — exécute le decision engine et journalise les décisions."""

from __future__ import annotations

import json
from pathlib import Path

import click
import mlflow
import pandas as pd

from cif_credit.config import load_config
from cif_credit.decision.engine import DecisionEngine, DecisionPolicy
from cif_credit.models.train import feature_columns
from cif_credit.utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--data", type=str, default=None, help="Chemin des features parquet (batch de décision).")
@click.option("--model-uri", type=str, default=None, help="URI MLflow du modèle (défaut: registry).")
@click.option("--out", type=str, default=None, help="Chemin de sortie JSON des décisions.")
def main(data: str | None, model_uri: str | None, out: str | None) -> None:
    """Applique le decision engine à un batch et trace chaque décision (audit §M08)."""
    cfg = load_config()
    mlflow.set_tracking_uri(cfg.model.mlflow_tracking_uri)
    path = data or f"{cfg.data.processed_dir}/{cfg.features.output_file}"
    df = pd.read_parquet(path)

    uri = model_uri or "models:/cif_credit_official/latest"
    model = mlflow.sklearn.load_model(uri)

    cols = feature_columns(cfg.features)
    probs = model.predict_proba(df[cols].astype(float))[:, 1]

    policy = DecisionPolicy(
        approve_threshold=cfg.decision.approve_threshold,
        review_threshold=cfg.decision.review_threshold,
        hard_reject_threshold=cfg.decision.hard_reject_threshold,
        thin_file_review=cfg.decision.thin_file_review,
        thin_file_max_loans=cfg.decision.thin_file_max_loans,
    )
    engine = DecisionEngine(model=model, policy=policy)

    n_loans = df["n_past_loans"].astype(int) if "n_past_loans" in df.columns else pd.Series(0, index=df.index)
    probs_df = pd.DataFrame({"probability": probs})
    outcomes = engine.decide_batch(probs=probs_df, n_past_loans=n_loans, customer_ids=df["customer_id"])

    decisions = pd.DataFrame([o.to_dict() for o in outcomes])
    distribution = decisions["decision"].value_counts().to_dict()

    out_path = Path(out or "data/artifacts/decisions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for o in outcomes:
            fh.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")

    logger.info("cli.decision.done", n_decisions=len(outcomes), distribution=distribution, path=str(out_path))
