"""Label-noise detectors for geospatial training data.

Every detector returns a *suspicion score* per sample: higher means the given
label is more likely to be wrong. Scores can be ranked for manual review or
thresholded to produce flags.

Detectors
---------
self_confidence            1 - out-of-fold probability of the given label
                           (the ranking used by confident learning).
neighbour_disagreement     share of the k nearest spatial neighbours whose
                           label differs (a purely spatial check).
smoothed_self_confidence   self-confidence computed on out-of-fold
                           probabilities that are averaged over spatial
                           neighbours (the method proposed in the paper).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from .spatial_cv import SpatialBlockKFold


def default_model(seed: int | None = 0) -> ClassifierMixin:
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed)


def out_of_fold_proba(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray | None = None,
    cv: str = "spatial",
    n_splits: int = 5,
    block_size: float = 10.0,
    model: ClassifierMixin | None = None,
    n_classes: int | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    """Predict class probabilities for every sample from a model that never saw it.

    ``cv="random"`` uses shuffled K-fold; ``cv="spatial"`` uses
    :class:`SpatialBlockKFold` and requires ``coords``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n_classes = int(n_classes if n_classes is not None else y.max() + 1)
    model = model if model is not None else default_model(seed)

    if cv == "random":
        splits = KFold(n_splits, shuffle=True, random_state=seed).split(X)
    elif cv == "spatial":
        if coords is None:
            raise ValueError("spatial cross-validation needs coords")
        splits = SpatialBlockKFold(n_splits, block_size, seed=seed).split(coords)
    else:
        raise ValueError(f"cv must be 'random' or 'spatial', got {cv!r}")

    proba = np.zeros((len(y), n_classes))
    for train, test in splits:
        m = clone(model).fit(X[train], y[train])
        proba[np.ix_(test, m.classes_)] = m.predict_proba(X[test])
    return proba


def self_confidence(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Suspicion = 1 - predicted probability of the given label."""
    return 1.0 - proba[np.arange(len(y)), y]


def confident_learning_flags(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Flag likely label errors with the confident-joint rule of Northcutt et al. (2021).

    Each class j gets a threshold t_j, the mean probability of class j among
    samples labelled j. A sample is flagged when some other class clears its own
    threshold and is the most probable such class.
    """
    n_classes = proba.shape[1]
    thresholds = np.array(
        [proba[y == j, j].mean() if np.any(y == j) else 1.0 for j in range(n_classes)]
    )
    above = proba >= thresholds
    masked = np.where(above, proba, -np.inf)
    confident_class = masked.argmax(axis=1)
    has_confident = above.any(axis=1)
    return has_confident & (confident_class != y)


def _neighbours(coords: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(coords) - 1)
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    return idx[:, 1:]  # drop self


def neighbour_disagreement(coords: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
    """Share of the k nearest spatial neighbours whose label differs."""
    nn = _neighbours(np.asarray(coords, dtype=float), k)
    return (y[nn] != y[:, None]).mean(axis=1)


def smooth_proba(proba: np.ndarray, coords: np.ndarray, k: int = 10, self_weight: float = 0.5) -> np.ndarray:
    """Blend each sample's probabilities with the mean of its k spatial neighbours."""
    nn = _neighbours(np.asarray(coords, dtype=float), k)
    return self_weight * proba + (1.0 - self_weight) * proba[nn].mean(axis=1)


def smoothed_self_confidence(
    proba: np.ndarray, y: np.ndarray, coords: np.ndarray, k: int = 10, self_weight: float = 0.5
) -> np.ndarray:
    """Self-confidence on spatially smoothed out-of-fold probabilities."""
    return self_confidence(smooth_proba(proba, coords, k, self_weight), y)


def rank_average(*scores: np.ndarray, weights: list[float] | None = None) -> np.ndarray:
    """Combine scores on different scales by averaging their normalised ranks."""
    weights = weights or [1.0] * len(scores)
    ranks = [w * rankdata(s) / len(s) for w, s in zip(weights, scores)]
    return np.sum(ranks, axis=0) / sum(weights)
