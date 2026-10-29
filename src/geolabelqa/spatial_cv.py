"""Spatial block cross-validation.

Random K-fold puts near neighbours on both sides of a split. On spatially
autocorrelated data that inflates accuracy, and - more importantly for label
auditing - it lets a model "explain" a wrong label by copying it from a nearby
training point that carries the same mistake. Holding out whole spatial
blocks removes that shortcut.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.spatial import cKDTree


class SpatialBlockKFold:
    """K-fold splitter that assigns square spatial blocks to folds.

    Parameters
    ----------
    n_splits:
        Number of folds.
    block_size:
        Side length of the square blocks, in coordinate units. Choose it larger
        than the range of spatial autocorrelation in the data.
    buffer:
        Optional distance. Training points closer than this to any test point
        are dropped from that fold's training set.
    seed:
        Seed for the random block-to-fold assignment.
    """

    def __init__(self, n_splits: int = 5, block_size: float = 10.0, buffer: float = 0.0, seed: int | None = None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.n_splits = n_splits
        self.block_size = block_size
        self.buffer = buffer
        self.seed = seed

    def block_ids(self, coords: np.ndarray) -> np.ndarray:
        """Integer id of the block that contains each point."""
        cells = np.floor((coords - coords.min(axis=0)) / self.block_size).astype(np.int64)
        _, ids = np.unique(cells, axis=0, return_inverse=True)
        return ids.ravel()

    def fold_ids(self, coords: np.ndarray) -> np.ndarray:
        """Fold index of each point, balancing point counts across folds."""
        blocks = self.block_ids(coords)
        counts = np.bincount(blocks)
        if (counts > 0).sum() < self.n_splits:
            raise ValueError(
                f"only {(counts > 0).sum()} non-empty blocks for {self.n_splits} folds; reduce block_size"
            )
        rng = np.random.default_rng(self.seed)
        # Greedy balancing: place blocks (largest first, ties shuffled) into the lightest fold.
        order = rng.permutation(len(counts))
        order = order[np.argsort(-counts[order], kind="stable")]
        load = np.zeros(self.n_splits)
        block_fold = np.empty(len(counts), dtype=int)
        for b in order:
            f = int(load.argmin())
            block_fold[b] = f
            load[f] += counts[b]
        return block_fold[blocks]

    def split(self, coords: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        coords = np.asarray(coords, dtype=float)
        folds = self.fold_ids(coords)
        tree = cKDTree(coords) if self.buffer > 0 else None
        for f in range(self.n_splits):
            test = np.flatnonzero(folds == f)
            train_mask = folds != f
            if tree is not None:
                near = tree.query_ball_point(coords[test], r=self.buffer)
                near_idx = np.unique(np.concatenate([np.asarray(n, dtype=int) for n in near]))
                train_mask[near_idx] = False
            yield np.flatnonzero(train_mask), test

    def get_n_splits(self) -> int:
        return self.n_splits
