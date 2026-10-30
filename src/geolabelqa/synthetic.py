"""Synthetic land-cover benchmark with controllable, spatially structured label noise.

The generator produces a spatially coherent land-cover map from smoothed
Gaussian random fields, samples point locations on it, and simulates
multispectral reflectance for each point. Label noise can then be injected
under three regimes that mirror how annotation errors arise in practice:

* ``uniform``   - random, independent mistakes (fatigue, mis-clicks).
* ``pair``      - systematic confusion between look-alike classes
                  (e.g. grassland vs cropland), independent in space.
* ``clustered`` - systematic confusion confined to spatial patches, as when
                  one annotator or one imagery tile is consistently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

CLASSES: tuple[str, ...] = ("water", "forest", "grassland", "cropland", "urban", "bare")
BANDS: tuple[str, ...] = ("blue", "green", "red", "nir", "swir1", "swir2")

# Mean surface reflectance per class (rows follow CLASSES, columns follow BANDS).
# Grassland/cropland and urban/bare are deliberately close to each other.
SIGNATURES = np.array(
    [
        [0.06, 0.05, 0.03, 0.02, 0.01, 0.01],  # water
        [0.03, 0.06, 0.04, 0.30, 0.15, 0.07],  # forest
        [0.05, 0.09, 0.07, 0.32, 0.25, 0.14],  # grassland
        [0.06, 0.10, 0.09, 0.35, 0.24, 0.16],  # cropland
        [0.10, 0.11, 0.12, 0.18, 0.22, 0.20],  # urban
        [0.14, 0.17, 0.21, 0.26, 0.33, 0.28],  # bare
    ]
)

# The class each class is most often mistaken for (used by "pair" and "clustered").
CONFUSION_PARTNER = np.array([1, 2, 3, 2, 5, 4])

NOISE_TYPES = ("uniform", "pair", "clustered")


@dataclass
class LandCoverDataset:
    """A labelled point sample drawn from a synthetic land-cover map."""

    X: np.ndarray  # (n, n_features) spectral bands + indices
    y: np.ndarray  # (n,) true class index
    coords: np.ndarray  # (n, 2) x/y in kilometres
    class_map: np.ndarray  # (grid, grid) true class per cell
    extent: float  # side length of the square study area, km
    feature_names: tuple[str, ...]

    @property
    def n_classes(self) -> int:
        return len(CLASSES)


def _class_map(rng: np.random.Generator, grid: int, length_scale: float) -> np.ndarray:
    fields = np.stack(
        [gaussian_filter(rng.standard_normal((grid, grid)), length_scale, mode="wrap") for _ in CLASSES]
    )
    fields /= fields.std(axis=(1, 2), keepdims=True)
    return fields.argmax(axis=0)


def _spectral_indices(refl: np.ndarray) -> np.ndarray:
    blue, green, red, nir, swir1, _ = refl.T
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndbi = (swir1 - nir) / (swir1 + nir + eps)
    return np.column_stack([ndvi, ndwi, ndbi])


def make_landcover(
    n_samples: int = 3000,
    extent: float = 50.0,
    grid: int = 200,
    length_scale: float = 6.0,
    spectral_noise: float = 0.30,
    illumination_strength: float = 0.15,
    seed: int | None = None,
) -> LandCoverDataset:
    """Generate a labelled point sample over a synthetic land-cover map.

    Parameters
    ----------
    n_samples:
        Number of labelled points.
    extent:
        Side of the square study area in kilometres.
    grid:
        Resolution of the underlying class map (cells per side).
    length_scale:
        Smoothing of the class random fields, in grid cells. Larger values give
        bigger homogeneous patches (stronger spatial autocorrelation).
    spectral_noise:
        Relative standard deviation of within-class reflectance variation.
    illumination_strength:
        Amplitude of a smooth multiplicative field that mimics atmospheric and
        illumination effects. It makes nearby samples look alike regardless of
        class - the reason random cross-validation is optimistic on spatial data.
    seed:
        Random seed.
    """
    rng = np.random.default_rng(seed)
    class_map = _class_map(rng, grid, length_scale)

    coords = rng.uniform(0.0, extent, size=(n_samples, 2))
    cells = np.clip((coords / extent * grid).astype(int), 0, grid - 1)
    y = class_map[cells[:, 1], cells[:, 0]]

    illum = gaussian_filter(rng.standard_normal((grid, grid)), length_scale * 2, mode="wrap")
    illum = 1.0 + illumination_strength * illum / illum.std()
    gain = illum[cells[:, 1], cells[:, 0]][:, None]

    mean = SIGNATURES[y]
    refl = mean * (1.0 + spectral_noise * rng.standard_normal(mean.shape)) * gain
    refl = np.clip(refl, 0.001, 1.0)

    X = np.column_stack([refl, _spectral_indices(refl)])
    names = BANDS + ("ndvi", "ndwi", "ndbi")
    return LandCoverDataset(X=X, y=y, coords=coords, class_map=class_map, extent=extent, feature_names=names)


def inject_noise(
    y: np.ndarray,
    coords: np.ndarray,
    noise_type: str,
    rate: float,
    n_classes: int = len(CLASSES),
    cluster_radius: float = 3.0,
    seed: int | None = None,
    partner: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt a fraction ``rate`` of labels under the given noise regime.

    ``partner[j]`` is the look-alike class that class ``j`` is relabelled to by
    the "pair" and "clustered" regimes (default: ``CONFUSION_PARTNER``).

    Returns ``(noisy_labels, is_noisy)``.
    """
    partner = CONFUSION_PARTNER if partner is None else np.asarray(partner)
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"noise_type must be one of {NOISE_TYPES}, got {noise_type!r}")
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must be in [0, 1)")

    rng = np.random.default_rng(seed)
    n = len(y)
    target = int(round(rate * n))
    noisy = y.copy()
    flip = np.zeros(n, dtype=bool)

    if noise_type == "uniform":
        flip[rng.choice(n, size=target, replace=False)] = True
        offsets = rng.integers(1, n_classes, size=target)
        noisy[flip] = (y[flip] + offsets) % n_classes

    elif noise_type == "pair":
        flip[rng.choice(n, size=target, replace=False)] = True
        noisy[flip] = partner[y[flip]]

    else:  # clustered
        # Grow circular "bad annotator" patches until the target rate is met.
        for centre in coords[rng.permutation(n)]:
            if flip.sum() >= target:
                break
            inside = np.hypot(*(coords - centre).T) <= cluster_radius
            new = np.flatnonzero(inside & ~flip)
            new = new[: target - flip.sum()]
            flip[new] = True
        noisy[flip] = partner[y[flip]]

    return noisy, flip
