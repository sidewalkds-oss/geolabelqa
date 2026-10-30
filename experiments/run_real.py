"""Benchmark label-noise detectors on real hyperspectral scenes with injected noise.

The scenes' expert ground truth is treated as clean. Noise is injected exactly
as in the synthetic benchmark, except that each class's look-alike partner is
estimated from data: the class it is most often confused with by a model
trained on clean labels.

Writes results/real_detection.csv and results/real_downstream.csv.

Usage:
    python experiments/run_real.py [--data data] [--seeds 5]
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from geolabelqa import detectors as d  # noqa: E402
from geolabelqa.metrics import detection_metrics, flag_metrics  # noqa: E402
from geolabelqa.scenes import SCENES, load_scene, most_confused_partner  # noqa: E402
from geolabelqa.synthetic import NOISE_TYPES, inject_noise  # noqa: E402
from run_benchmark import rf  # noqa: E402

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

N_TRAIN, N_TEST = 3000, 1000
RATE = 0.2
K = 10


def run(scene, seed, partner, noise_type):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(scene.y), size=N_TRAIN + N_TEST, replace=False)
    tr, te = idx[:N_TRAIN], idx[N_TRAIN:]
    coords, y_true = scene.coords[tr], scene.y[tr]
    n_classes = len(scene.class_ids)
    # Error patches and CV blocks scale with the scene: radius = extent/15, block = extent/5.
    y_noisy, is_noisy = inject_noise(y_true, coords, noise_type, RATE, n_classes=n_classes,
                                     cluster_radius=scene.extent / 15, seed=seed + 1000, partner=partner)
    base = dict(scene=scene.name, noise=noise_type, rate=RATE, seed=seed)

    det, down = [], []
    nbr = d.neighbour_disagreement(coords, y_noisy, K)
    det.append({**base, "features": "-", "cv": "-", "detector": "neighbour_vote", **detection_metrics(is_noisy, nbr)})
    rankings = {("neighbour_vote", "-", "-"): nbr}

    def feats(i, fs):
        return np.column_stack([scene.X[i], scene.coords[i]]) if fs == "spectral+xy" else scene.X[i]

    for fs, cv in itertools.product(("spectral", "spectral+xy"), ("random", "spatial")):
        p = d.out_of_fold_proba(feats(tr, fs), y_noisy, coords, cv=cv, block_size=scene.extent / 5,
                                model=rf(seed), n_classes=n_classes, seed=seed)
        for name, proba in (("self_confidence", p), ("smoothed_sc", d.smooth_proba(p, coords, K))):
            score = d.self_confidence(proba, y_noisy)
            flags = flag_metrics(is_noisy, d.confident_learning_flags(proba, y_noisy))
            det.append({**base, "features": fs, "cv": cv, "detector": name, **detection_metrics(is_noisy, score),
                        **{f"cl_{k}": v for k, v in flags.items()}})
            rankings[(name, fs, cv)] = score

    fs = "spectral"
    n_noisy = int(is_noisy.sum())

    def evaluate(method, keep, labels):
        pred = rf(seed).fit(feats(tr[keep], fs), labels[keep]).predict(feats(te, fs))
        down.append({**base, "features": fs, "method": method,
                     "accuracy": float((pred == scene.y[te]).mean()),
                     "macro_f1": f1_score(scene.y[te], pred, average="macro")})

    everything = np.ones(N_TRAIN, dtype=bool)
    evaluate("no_cleaning", everything, y_noisy)
    evaluate("oracle_remove_noisy", ~is_noisy, y_noisy)
    evaluate("oracle_clean_labels", everything, y_true)
    for (name, rfs, cv), score in rankings.items():
        if rfs in ("-", fs):
            keep = everything.copy()
            keep[np.argsort(-score, kind="stable")[:n_noisy]] = False
            evaluate(f"remove:{name}|{cv}", keep, y_noisy)
    return det, down


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det_path, down_path = out / "real_detection.csv", out / "real_downstream.csv"
    det = pd.read_csv(det_path) if det_path.exists() else pd.DataFrame()
    down = pd.read_csv(down_path) if down_path.exists() else pd.DataFrame()
    done = set() if det.empty else set(map(tuple, det[["scene", "noise", "seed"]].drop_duplicates().to_numpy().tolist()))

    t0 = time.time()
    for name in SCENES:
        scene = load_scene(name, args.data)
        # Estimate look-alike pairs once per scene from a clean-label sample not otherwise used.
        rng = np.random.default_rng(999)
        idx = rng.choice(len(scene.y), size=N_TRAIN, replace=False)
        p = d.out_of_fold_proba(scene.X[idx], scene.y[idx], cv="random", model=rf(999),
                                n_classes=len(scene.class_ids), seed=999)
        partner = most_confused_partner(p, scene.y[idx])
        print(f"{name}: {len(scene.y)} labelled px, {len(scene.class_ids)} classes, "
              f"clean OOF acc {np.mean(p.argmax(1) == scene.y[idx]):.3f}, partner {partner.tolist()}", flush=True)

        for noise, seed in itertools.product(NOISE_TYPES, range(args.seeds)):
            if (name, noise, seed) in done:
                continue
            rows, drows = run(scene, seed, partner, noise)
            det = pd.concat([det, pd.DataFrame(rows)], ignore_index=True)
            down = pd.concat([down, pd.DataFrame(drows)], ignore_index=True)
            det.to_csv(det_path, index=False)
            down.to_csv(down_path, index=False)
            print(f"  {noise:9s} seed={seed}  {time.time() - t0:6.0f}s", flush=True)
        del scene
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
