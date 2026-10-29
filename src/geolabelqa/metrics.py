"""Metrics for scoring a label-noise detector against known corrupted labels."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def detection_metrics(is_noisy: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Ranking quality of ``scores`` for recovering ``is_noisy``.

    ``precision_at_n`` reviews exactly as many samples as there are true errors
    (so it equals recall at that budget) - a realistic "fix the top-N" workflow.
    """
    is_noisy = np.asarray(is_noisy, dtype=bool)
    n_noisy = int(is_noisy.sum())
    if n_noisy == 0 or n_noisy == len(is_noisy):
        return {"auroc": float("nan"), "ap": float("nan"), "precision_at_n": float("nan")}
    top = np.argsort(-scores, kind="stable")[:n_noisy]
    return {
        "auroc": float(roc_auc_score(is_noisy, scores)),
        "ap": float(average_precision_score(is_noisy, scores)),
        "precision_at_n": float(is_noisy[top].mean()),
    }


def flag_metrics(is_noisy: np.ndarray, flags: np.ndarray) -> dict[str, float]:
    """Precision, recall and F1 of a binary flag set."""
    is_noisy = np.asarray(is_noisy, dtype=bool)
    flags = np.asarray(flags, dtype=bool)
    tp = int((is_noisy & flags).sum())
    precision = tp / flags.sum() if flags.any() else 0.0
    recall = tp / is_noisy.sum() if is_noisy.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n_flagged": int(flags.sum())}
