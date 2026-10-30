"""Clean-label out-of-fold accuracy under random vs spatial block CV.

The gap measures how much a model depends on seeing the same neighbourhood
during training, i.e. how predictable a held-out block is from the rest. It
explains when spatial CV helps label auditing and when it does not.

Writes results/transferability.csv.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from geolabelqa import detectors as d  # noqa: E402
from geolabelqa.scenes import SCENES, load_scene  # noqa: E402
from geolabelqa.synthetic import make_landcover  # noqa: E402
from run_benchmark import rf  # noqa: E402

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
N, SEEDS = 3000, range(3)


def accuracy(X, y, coords, cv, block, n_classes, seed):
    p = d.out_of_fold_proba(X, y, coords, cv=cv, block_size=block, model=rf(seed), n_classes=n_classes, seed=seed)
    return float((p.argmax(1) == y).mean())


def main():
    rows = []
    for seed in SEEDS:
        ds = make_landcover(n_samples=N, seed=seed)
        for cv in ("random", "spatial"):
            rows.append({"dataset": "synthetic", "seed": seed, "cv": cv,
                         "accuracy": accuracy(ds.X, ds.y, ds.coords, cv, 10.0, 6, seed)})
    for name in SCENES:
        scene = load_scene(name, ROOT / "data")
        for seed in SEEDS:
            idx = np.random.default_rng(seed).choice(len(scene.y), size=N, replace=False)
            for cv in ("random", "spatial"):
                rows.append({"dataset": name, "seed": seed, "cv": cv,
                             "accuracy": accuracy(scene.X[idx], scene.y[idx], scene.coords[idx], cv,
                                                  scene.extent / 5, len(scene.class_ids), seed)})
        del scene
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "transferability.csv", index=False)
    print(df.groupby(["dataset", "cv"]).accuracy.agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
