"""Définitions Dagster — orchestration des assets du projet CIF Credit Intelligence.

Assets (versionnés, lineage automatique) :
- generate_raw_data  → génère les trois tables
- build_features     → 25 features + target
- train_model        → entraîne XGBoost, journalise dans MLflow, enregistre dans le registry
- evaluate_model     → bootstrap, robustesse, fairness, ablation, GO/NO-GO
- decide_batch       → applique le decision engine (human-in-the-loop)

Le passage des assets est strictement séquentiel ; chaque asset est reproductible
(seed fixe, config Hydra versionnée).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import dagster as dg
import mlflow
import pandas as pd

from config import load_config
from data.synthetic import generate_datasets, save_datasets
from evaluation.ablation import run_ablation
from evaluation.bootstrap import bootstrap_metrics
from evaluation.fairness import fairness_by_group
from evaluation.robustness import robustness_curve
from features.builder import build_features as build_features_df
from models.train import feature_columns, train_and_log
from utils.logging import get_logger

logger = get_logger(__name__)
CONF = load_config()


def _mlflow_env() -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", CONF.model.mlflow_tracking_uri))
    mlflow.set_experiment(CONF.model.mlflow_experiment)


@dg.asset(group_name="data", compute_kind="python")
def generate_raw_data(context: dg.AssetExecutionContext) -> None:
    """Génère les données synthétiques (10k clients / ~25k prêts / 240k relevés)."""
    datasets = generate_datasets(CONF.data)
    paths = save_datasets(CONF.data, datasets)
    context.log.info("raw data générées", extra={"paths": paths})


@dg.asset(group_name="data", deps=["generate_raw_data"], compute_kind="python")
def build_features(context: dg.AssetExecutionContext) -> None:
    """Construit le jeu de 25 features + target."""
    raw_dir = Path(CONF.data.raw_dir)
    customers = pd.read_parquet(raw_dir / "customers.parquet")
    loans = pd.read_parquet(raw_dir / "loans.parquet")
    savings = pd.read_parquet(raw_dir / "savings.parquet")
    features = build_features_df(customers, loans, savings, CONF.features)
    out = Path(CONF.data.processed_dir) / CONF.features.output_file
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    context.log.info("features construites", extra={"n": len(features), "path": str(out)})


@dg.asset(group_name="models", deps=["build_features"], compute_kind="xgboost")
def train_model(context: dg.AssetExecutionContext) -> None:
    """Entraîne le modèle officiel et l'enregistre dans le registry MLflow."""
    _mlflow_env()
    df = pd.read_parquet(Path(CONF.data.processed_dir) / CONF.features.output_file)
    result = train_and_log(df, CONF.model, CONF.features, register=True, run_name="dagster-xgboost")
    context.log.info("modèle entraîné", extra={"run_id": result.run_id, "metrics": result.metrics})


@dg.asset(group_name="models", deps=["train_model"], compute_kind="python")
def evaluate_model(context: dg.AssetExecutionContext) -> None:
    """Audit : bootstrap, robustesse, fairness, ablation puis verdict GO/NO-GO."""
    df = pd.read_parquet(Path(CONF.data.processed_dir) / CONF.features.output_file)
    cols = feature_columns(CONF.features)
    X = df[cols].astype(float)
    y = df[CONF.features.target].astype(int).to_numpy()

    report_dir = Path(CONF.evaluation.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import train_test_split

    X_te, _, y_te, _ = train_test_split(
        X, y, test_size=CONF.model.test_size, random_state=CONF.model.random_state, stratify=y
    )

    from models.train import train_plain

    model, base_metrics = train_plain(df, CONF.model, CONF.features)
    probs = model.predict_proba(X_te)[:, 1]

    final: dict[str, Any] = {"base_metrics": base_metrics}
    final["bootstrap"] = bootstrap_metrics(y_te, probs, CONF.evaluation.bootstrap_iterations, CONF.model.random_state)
    final["robustness"] = robustness_curve(
        model, X_te, y_te, CONF.evaluation.robustness_noise_levels, CONF.model.random_state
    )
    final["ablation"] = run_ablation(df, CONF.model, CONF.features, seed=CONF.model.random_state)
    groups = next((c for c in ["gender", "sector", "location"] if c in df.columns), None)
    if groups:
        final["fairness"] = fairness_by_group(df.loc[X_te.index], y_te, probs, groups)

    roc = float(base_metrics["roc_auc"])
    ece = float(base_metrics["ece"])
    decision_gate = "GO" if (roc >= CONF.evaluation.go_roc_auc and ece <= CONF.evaluation.ece_threshold) else "NO-GO"
    final["decision_gate"] = decision_gate

    (report_dir / "evaluation_report.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    context.log.info("évaluation terminée", extra={"gate": final["decision_gate"]})


@dg.asset(group_name="decisions", deps=["train_model"], compute_kind="python")
def decide_batch(context: dg.AssetExecutionContext) -> None:
    """Applique le decision engine au batch complet et trace chaque décision."""
    out = Path("data/artifacts") / "decisions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    from services.decision_engine import DecisionEngine, DecisionPolicy

    df = pd.read_parquet(Path(CONF.data.processed_dir) / CONF.features.output_file)
    _mlflow_env()
    model = mlflow.sklearn.load_model("models:/cif_credit_official/latest")
    cols = feature_columns(CONF.features)
    probs = model.predict_proba(df[cols].astype(float))[:, 1]

    policy = DecisionPolicy(
        approve_threshold=CONF.decision.approve_threshold,
        review_threshold=CONF.decision.review_threshold,
        hard_reject_threshold=CONF.decision.hard_reject_threshold,
        thin_file_review=CONF.decision.thin_file_review,
        thin_file_max_loans=CONF.decision.thin_file_max_loans,
    )
    engine = DecisionEngine(model=model, policy=policy)
    n_loans = df["n_past_loans"].astype(int) if "n_past_loans" in df.columns else pd.Series(0, index=df.index)
    outcomes = engine.decide_batch(
        probs=pd.DataFrame({"probability": probs}), n_past_loans=n_loans, customer_ids=df["customer_id"]
    )
    with out.open("w", encoding="utf-8") as fh:
        for o in outcomes:
            fh.write(json.dumps(o.to_dict()) + "\n")
    context.log.info("décisions appliquées", extra={"n": len(outcomes), "path": str(out)})


single_asset_job = dg.define_asset_job(
    name="cif_full_pipeline",
    selection=[generate_raw_data, build_features, train_model, evaluate_model, decide_batch],
)

daily_schedule = dg.ScheduleDefinition(
    job=single_asset_job,
    cron_schedule="0 2 * * *",  # quotidien à 02:00
    execution_timezone="Africa/Dakar",
)

defs = dg.Definitions(
    assets=[
        generate_raw_data,
        build_features,
        train_model,
        evaluate_model,
        decide_batch,
    ],
    jobs=[single_asset_job],
    schedules=[daily_schedule],
    resources={},
)
