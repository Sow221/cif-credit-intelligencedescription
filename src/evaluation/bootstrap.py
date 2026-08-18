"""Bootstrap stratifié et intervalles de confiance — protocole CIF (1000 réplications)."""

from __future__ import annotations

import numpy as np

from evaluation.metrics import compute_all_metrics
from utils.logging import get_logger

logger = get_logger(__name__)


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 1_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Calcule moyennes + IC95% (percentiles 2.5/97.5) des métriques par bootstrap stratifié.

    La stratification préserve la proportion de défauts dans chaque réplication (règle du
    protocole : « Stratifié — préserver la proportion de défauts »).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    metric_names = ["roc_auc", "pr_auc", "brier", "ece"]
    samples: dict[str, list[float]] = {m: [] for m in metric_names}

    for _ in range(n_iterations):
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            break
        pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pos_sample, neg_sample])
        rng.shuffle(idx)

        metrics = compute_all_metrics(y_true[idx], y_prob[idx])
        for m in metric_names:
            samples[m].append(metrics[m])

    result: dict[str, dict[str, float]] = {}
    for m, vals in samples.items():
        if not vals:
            continue
        arr = np.asarray(vals)
        result[m] = {
            "mean": float(arr.mean()),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "std": float(arr.std()),
        }
    logger.info("evaluation.bootstrap.done", n_iterations=len(next(iter(samples.values()))), metrics=result)
    return result
