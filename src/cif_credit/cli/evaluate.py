"""CLI : cif-evaluate — audit complet (Phase B/C : bootstrap, robustesse, fairness, ablation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from cif_credit.config import load_config
from cif_credit.evaluation.ablation import run_ablation
from cif_credit.evaluation.bootstrap import bootstrap_metrics
from cif_credit.evaluation.fairness import fairness_by_group
from cif_credit.evaluation.robustness import robustness_curve
from cif_credit.models.train import feature_columns, train_plain
from cif_credit.utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--data", type=str, default=None, help="Chemin des features parquet.")
@click.option("--bootstrap/--no-bootstrap", default=True, help="Calcule le bootstrap IC95%.")
@click.option("--robustness/--no-robustness", default=True, help="Courbe de robustesse au bruit.")
@click.option("--fairness/--no-fairness", default=True, help="Analyse fairness par groupe.")
@click.option("--ablation/--no-ablation", default=True, help="Ablation par famille de features.")
def main(
    data: str | None,
    bootstrap: bool,
    robustness: bool,
    fairness: bool,
    ablation: bool,
) -> None:
    """Exécute le harness d'audit et écrit les rapports JSON sous data/artifacts/reports."""
    cfg = load_config()
    path = data or f"{cfg.data.processed_dir}/{cfg.features.output_file}"
    df = pd.read_parquet(path)

    report_dir = Path(cfg.evaluation.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    cols = feature_columns(cfg.features)
    X = df[cols].astype(float)
    y = df[cfg.features.target].astype(int).to_numpy()
    _X_tr, X_te, _y_tr, y_te = train_test_split(
        X, y, test_size=cfg.model.test_size, random_state=cfg.model.random_state, stratify=y
    )

    final: dict[str, Any] = {"n_samples": len(df), "n_features": len(cols)}

    model, base_metrics = train_plain(df, cfg.model, cfg.features)
    final["base_metrics"] = base_metrics
    probs = model.predict_proba(X_te)[:, 1]

    if bootstrap:
        final["bootstrap"] = bootstrap_metrics(
            y_te, probs, n_iterations=cfg.evaluation.bootstrap_iterations, seed=cfg.model.random_state
        )

    if robustness:
        final["robustness"] = robustness_curve(
            model, X_te, y_te, cfg.evaluation.robustness_noise_levels, seed=cfg.model.random_state
        )

    if fairness:
        groups = next(
            (c for c in ["gender", "sector", "location"] if c in df.columns),
            None,
        )
        if groups:
            final["fairness"] = fairness_by_group(df.loc[X_te.index], y_te, probs, groups)

    if ablation:
        final["ablation"] = run_ablation(df, cfg.model, cfg.features, seed=cfg.model.random_state)

    # Verdict GO/NO-GO selon le protocole CIF
    roc = float(final["base_metrics"]["roc_auc"])
    ece = float(final["base_metrics"]["ece"])
    decision_gate = "GO" if (roc >= cfg.evaluation.go_roc_auc and ece <= cfg.evaluation.ece_threshold) else "NO-GO"
    if roc < cfg.evaluation.nogo_roc_auc or ece > 2 * cfg.evaluation.ece_threshold:
        decision_gate = "NO-GO"
    final["decision_gate"] = decision_gate

    out_path = report_dir / "evaluation_report.json"
    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("cli.evaluate.done", path=str(out_path), decision_gate=decision_gate, roc_auc=roc)
