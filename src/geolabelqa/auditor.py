"""High-level API: audit a table of labelled, geolocated samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from . import detectors as d


class LabelAuditor:
    """Rank geolocated training samples by how likely their label is wrong.

    The default pipeline is the one recommended by the accompanying paper:
    out-of-fold probabilities from spatial block cross-validation, smoothed over
    each sample's spatial neighbours, scored by self-confidence. Confident-learning
    flags and a neighbour-disagreement score are reported alongside.

    Parameters
    ----------
    block_size:
        Side of the cross-validation blocks, in coordinate units. Use a projected
        CRS (metres/kilometres), not degrees.
    k:
        Number of spatial neighbours used for smoothing and disagreement.
    self_weight:
        Weight of a sample's own prediction when smoothing (1.0 disables smoothing).
    cv:
        ``"spatial"`` (default) or ``"random"``.
    """

    def __init__(
        self,
        block_size: float = 10.0,
        k: int = 10,
        self_weight: float = 0.5,
        cv: str = "spatial",
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
