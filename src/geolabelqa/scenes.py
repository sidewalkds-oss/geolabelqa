"""Loaders for public hyperspectral benchmark scenes with ground-truth maps.

The scenes are distributed by the Computational Intelligence Group of the
University of the Basque Country (UPV/EHU):
https://ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

Download the MATLAB files listed in ``SCENES`` into a folder (default ``data/``).
Their ground-truth maps are expert-labelled and spatially coherent (whole fields,
roofs, roads), which makes them a realistic substrate for spatially structured
label noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SOURCE_URL = "https://ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes"

SCENES = {
    # name: (cube file, ground-truth file, ground sampling distance in metres)
    "indian_pines": ("Indian_pines_corrected.mat", "Indian_pines_gt.mat", 20.0),
    "pavia_university": ("PaviaU.mat", "PaviaU_gt.mat", 1.3),
}


@dataclass
class Scene:
    name: str
    X: np.ndarray  # (n, bands) reflectance of labelled pixels, float32
    y: np.ndarray  # (n,) class index 0..K-1 (remapped)
    coords: np.ndarray  # (n, 2) easting/northing in metres from the image origin
    class_ids: np.ndarray  # original ground-truth codes, indexed by y
    shape: tuple[int, int]
    gsd: float

    @property
    def extent(self) -> float:
        return max(self.shape) * self.gsd


def _array(mat: dict) -> np.ndarray:
    arrays = [v for k, v in mat.items() if not k.startswith("__") and isinstance(v, np.ndarray)]
    if len(arrays) != 1:
        raise ValueError(f"expected one array in .mat file, found {len(arrays)}")
    return arrays[0]


def load_scene(name: str, data_dir: str | Path = "data", min_class_size: int = 400) -> Scene:
    """Load the labelled pixels of a scene.

    Classes with fewer than ``min_class_size`` labelled pixels are dropped. A
    tiny class confined to one field cannot be predicted when spatial CV holds
    that field out, and would only add noise to the comparison.
    """
    cube_file, gt_file, gsd = SCENES[name]
    data_dir = Path(data_dir)
    missing = [f for f in (cube_file, gt_file) if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in {data_dir}; download them from {SOURCE_URL}")

    cube = _array(loadmat(data_dir / cube_file)).astype(np.float32)
    gt = _array(loadmat(data_dir / gt_file)).astype(int)
    if cube.shape[:2] != gt.shape:
        raise ValueError(f"cube {cube.shape} and ground truth {gt.shape} do not align")

    codes, counts = np.unique(gt[gt > 0], return_counts=True)
    keep = codes[counts >= min_class_size]
    rows, cols = np.nonzero(np.isin(gt, keep))
    labels = gt[rows, cols]
    y = np.searchsorted(keep, labels)

    X = cube[rows, cols]
    X /= X.max()  # scenes ship in different integer scalings
    coords = np.column_stack([cols * gsd, (gt.shape[0] - 1 - rows) * gsd]).astype(float)
    return Scene(name, X, y, coords, keep, gt.shape, gsd)


def most_confused_partner(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """For each class, the other class it is most often predicted as.

    Used to build realistic class-pair and clustered noise on real scenes, where
    the look-alike pairs are not known in advance.
    """
    n_classes = proba.shape[1]
    confusion = np.zeros((n_classes, n_classes))
    for j in range(n_classes):
        confusion[j] = proba[y == j].mean(axis=0)
    np.fill_diagonal(confusion, -np.inf)
    return confusion.argmax(axis=1)
