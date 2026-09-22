"""Définitions Dagster — orchestration des assets du projet CIF Credit Intelligence.

Point d'entrée unique de visibilité sur l'état des données et des modèles (interface web,
``dagster dev``) : deux graphes d'assets, plutôt que des scripts isolés invisibles les uns
des autres.

Groupe ``synthetic`` (prototype d'infrastructure) :
- generate_raw_data → build_features → train_model → evaluate_model → decide_batch

Groupe ``lending_club`` (validation de la méthode sur données publiques réelles) :
- lc_raw_available → lc_interim → lc_features → lc_benchmark

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
def generate_raw_data(context) -> None:
    """Génère les données synthétiques (10k clients / ~25k prêts / 240k relevés)."""
    datasets = generate_datasets(CONF.data)
    paths = save_datasets(CONF.data, datasets)
    context.log.info("raw data générées", extra={"paths": paths})


@dg.asset(group_name="data", deps=["generate_raw_data"], compute_kind="python")
def build_features(context) -> None:
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
def train_model(context) -> None:
    """Entraîne le modèle officiel et l'enregistre dans le registry MLflow."""
    _mlflow_env()
    df = pd.read_parquet(Path(CONF.data.processed_dir) / CONF.features.output_file)
    result = train_and_log(df, CONF.model, CONF.features, register=True, run_name="dagster-xgboost")
    context.log.info("modèle entraîné", extra={"run_id": result.run_id, "metrics": result.metrics})


@dg.asset(group_name="models", deps=["train_model"], compute_kind="python")
def evaluate_model(context) -> None:
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
def decide_batch(context) -> None:
    """Applique le decision engine au batch complet et trace chaque décision."""
    out = Path("data/artifacts") / "decisions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    from services.decision_engine import DecisionEngine, DecisionPolicy

    df = pd.read_parquet(Path(CONF.data.processed_dir) / CONF.features.output_file)
    _mlflow_env()
    model = mlflow.sklearn.load_model("models:/cif_credit_official@champion")
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


# --- Lending Club : validation de la méthode sur données publiques réelles ---
# Mêmes fonctions que les CLI `cif-ingest-lc` / `cif-benchmark` (aucune logique dupliquée) :
# Dagster n'est ici qu'une couche de visibilité et d'orchestration au-dessus du même code.


@dg.asset(group_name="lending_club", compute_kind="python")
def lc_raw_available(context) -> dg.MaterializeResult:
    """Vérifie la présence du fichier brut (téléchargé manuellement, jamais généré : voir `make data-download`)."""
    from data.lending_club import find_raw_file

    path = find_raw_file(CONF.lending_club.raw_dir)
    size_mb = path.stat().st_size / 1e6
    context.log.info("lending_club.raw.found", extra={"path": str(path)})
    return dg.MaterializeResult(metadata={"path": str(path), "size_mb": round(size_mb, 1)})


@dg.asset(group_name="lending_club", deps=["lc_raw_available"], compute_kind="pandera")
def lc_interim(context) -> dg.MaterializeResult:
    """Nettoie et valide (contrat Pandera bloquant) : raw → interim."""
    from data.lending_club import DATE_COL, TARGET, build_interim

    path = build_interim(CONF.lending_club)
    df = pd.read_parquet(path)
    context.log.info("lending_club.interim.done", extra={"rows": len(df)})
    return dg.MaterializeResult(
        metadata={
            "rows": len(df),
            "default_rate": round(float(df[TARGET].mean()), 4),
            "period": f"{df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}",
            "path": str(path),
        }
    )


@dg.asset(group_name="lending_club", deps=["lc_interim"], compute_kind="python")
def lc_features(context) -> dg.MaterializeResult:
    """Features point-in-time (sans fuite d'après-octroi) : interim → processed."""
    from features.lending_club import build_features as build_lc_features

    interim = pd.read_parquet(CONF.lending_club.interim_path)
    features = build_lc_features(interim)
    out = Path(CONF.lending_club.processed_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    context.log.info("lending_club.features.done", extra={"rows": len(features)})
    return dg.MaterializeResult(metadata={"rows": len(features), "n_features": len(features.columns) - 3})


@dg.asset(group_name="lending_club", deps=["lc_features"], compute_kind="xgboost")
def lc_benchmark(context) -> dg.MaterializeResult:
    """Baseline logistique vs XGBoost contraint, protocole out-of-time (ADR 0001-0003)."""
    from models.lending_club_registry import register_champion
    from pipelines.benchmark import run_benchmark, write_artifacts

    features = pd.read_parquet(CONF.lending_club.processed_path)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", CONF.model.mlflow_tracking_uri))
    mlflow.set_experiment("lending_club_benchmark")
    with mlflow.start_run(run_name="dagster-benchmark"):
        result = run_benchmark(features, CONF.lending_club)
        out = write_artifacts(result, "reports/lending_club")
        for name, metrics in result.metrics["test_metrics"].items():
            mlflow.log_metrics({f"test_{name}_{k}": v for k, v in metrics.items()})
        mlflow.log_artifacts(str(out))

        champ_metrics = result.metrics["test_metrics"][result.champion.name]
        mlflow.log_metrics({"roc_auc": champ_metrics["roc_auc"], "ece": champ_metrics["ece"]})
        version = register_champion(result.champion, result.test)
        context.log.info("lending_club.registry.done", extra={"version": version})
    m = result.metrics["test_metrics"]
    context.log.info("lending_club.benchmark.done", extra={"champion": result.metrics["champion"]})
    return dg.MaterializeResult(
        metadata={
            "champion": result.metrics["champion"],
            "reason": result.metrics["champion_reason"],
            "roc_auc_logistic": round(m["logistic"]["roc_auc"], 4),
            "roc_auc_xgboost": round(m["xgboost"]["roc_auc"], 4),
            "registered_version": version,
            "report": dg.MetadataValue.path(str(out / "metrics.json")),
        }
    )


lending_club_job = dg.define_asset_job(
    name="lending_club_pipeline",
    selection=[lc_raw_available, lc_interim, lc_features, lc_benchmark],
)

defs = dg.Definitions(
    assets=[
        generate_raw_data,
        build_features,
        train_model,
        evaluate_model,
        decide_batch,
        lc_raw_available,
        lc_interim,
        lc_features,
        lc_benchmark,
    ],
    jobs=[single_asset_job, lending_club_job],
    schedules=[daily_schedule],
    resources={},
)
