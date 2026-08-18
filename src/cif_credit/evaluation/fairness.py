"""Analyse de fairness par groupe (audit Phase C) — métriques + IC95% par groupe."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cif_credit.evaluation.metrics import compute_all_metrics


def fairness_by_group(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_col: str,
) -> dict[str, dict[str, str | float]]:
    """Calcule ROC-AUC, Brier et PR-AUC par groupe, avec comptage d'effectifs.

    Règle du protocole CIF : les petits groupes sont marqués « estimation instable »
    (représenté ici par n < 50).
    """
    result: dict[str, dict[str, str | float]] = {}
    for group, idx in df.groupby(group_col).groups.items():
        y_g = y_true[idx.to_numpy()]
        p_g = y_prob[idx.to_numpy()]
        if len(np.unique(y_g)) < 2:
            result[str(group)] = {"n": len(y_g), "roc_auc": float("nan"), "warning": "classe unique"}
            continue
        metrics = compute_all_metrics(y_g, p_g)
        entry: dict[str, str | float] = {
            "n": float(len(y_g)),
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "brier": metrics["brier"],
        }
        if len(y_g) < 50:
            entry["warning"] = 1.0  # estimation instable
        result[str(group)] = entry
    return result
