"""High-level API: audit a table of labelled, geolocated samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from . import detectors as d


class LabelAuditor:
    """Rank geolocated training samples by how likely their label is wrong.

    The pipeline follows the accompanying paper: out-of-fold class probabilities,
    smoothed over each sample's spatial neighbours, scored by self-confidence.
    Confident-learning flags and a neighbour-disagreement score are reported
    alongside.

    Spatial block folds help only when held-out blocks stay predictable from
    the rest of the data. Use :meth:`transferability_gap` to check: if the gap
    is small (a few points of accuracy), pass ``cv="spatial"``; otherwise keep
    the default random folds.

    Parameters
    ----------
    block_size:
        Side of the cross-validation blocks, in coordinate units. Use a projected
        CRS (metres/kilometres), not degrees. Only used with ``cv="spatial"``.
    k:
        Number of spatial neighbours used for smoothing and disagreement.
    self_weight:
        Weight of a sample's own prediction when smoothing (1.0 disables smoothing).
    cv:
        ``"random"`` (default) or ``"spatial"``.
    """

    def __init__(
        self,
        block_size: float = 10.0,
        k: int = 10,
        self_weight: float = 0.5,
        cv: str = "random",
        n_splits: int = 5,
        model: ClassifierMixin | None = None,
        seed: int | None = 0,
    ):
        self.block_size = block_size
        self.k = k
        self.self_weight = self_weight
        self.cv = cv
        self.n_splits = n_splits
        self.model = model
        self.seed = seed

    def transferability_gap(self, X, labels, coords) -> dict[str, float]:
        """Out-of-fold accuracy on the given labels under random and spatial folds.

        A large ``gap`` means held-out blocks are hard to predict, and spatial
        folds would starve the audit; a small one means spatial folds are safe.
        """
        X = np.asarray(X, dtype=float)
        coords = np.asarray(coords, dtype=float)
        classes, y = np.unique(np.asarray(labels), return_inverse=True)
        acc = {}
        for cv in ("random", "spatial"):
            proba = d.out_of_fold_proba(
                X, y, coords, cv=cv, n_splits=self.n_splits, block_size=self.block_size,
                model=self.model, n_classes=len(classes), seed=self.seed,
            )
            acc[cv] = float((proba.argmax(axis=1) == y).mean())
        return {"random": acc["random"], "spatial": acc["spatial"], "gap": acc["random"] - acc["spatial"]}

    def audit(self, X, labels, coords) -> pd.DataFrame:
        """Return one row per sample, sorted from most to least suspicious.

        Columns: ``index`` (original position), ``label``, ``suggested_label``,
        ``suspicion`` (0-1), ``cl_flag``, ``neighbour_disagreement``.
        """
        X = np.asarray(X, dtype=float)
        coords = np.asarray(coords, dtype=float)
        classes, y = np.unique(np.asarray(labels), return_inverse=True)

        proba = d.out_of_fold_proba(
            X, y, coords, cv=self.cv, n_splits=self.n_splits, block_size=self.block_size,
            model=self.model, n_classes=len(classes), seed=self.seed,
        )
        smoothed = d.smooth_proba(proba, coords, self.k, self.self_weight)
        report = pd.DataFrame(
            {
                "index": np.arange(len(y)),
                "label": classes[y],
                "suggested_label": classes[smoothed.argmax(axis=1)],
                "suspicion": d.self_confidence(smoothed, y),
                "cl_flag": d.confident_learning_flags(smoothed, y),
                "neighbour_disagreement": d.neighbour_disagreement(coords, y, self.k),
            }
        )
        return report.sort_values("suspicion", ascending=False, kind="stable").reset_index(drop=True)
