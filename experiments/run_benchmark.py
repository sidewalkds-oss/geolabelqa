"""Benchmark label-noise detectors on the synthetic land-cover task.

Writes:
    results/detection.csv   one row per (config, feature set, CV, detector)
    results/downstream.csv  test accuracy after removing the top-ranked samples
    results/ablation_smoothing.csv
    results/ablation_block_size.csv

Usage:
    python experiments/run_benchmark.py              # full run (~20 min on 8 cores)
    python experiments/run_benchmark.py --quick      # 1 seed, smoke test
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geolabelqa import detectors as d  # noqa: E402
from geolabelqa.metrics import detection_metrics, flag_metrics  # noqa: E402
from geolabelqa.synthetic import NOISE_TYPES, inject_noise, make_landcover  # noqa: E402

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

N_TRAIN, N_TEST = 3000, 1000
RATES = (0.1, 0.2, 0.3)
BLOCK_SIZE = 10.0  # km; study area is 50 x 50 km
K = 10


def rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=2, random_state=seed)


def features(ds, idx, feature_set):
    X = ds.X[idx]
    return np.column_stack([X, ds.coords[idx]]) if feature_set == "spectral+xy" else X


def run_config(noise_type, rate, seed):
    ds = make_landcover(n_samples=N_TRAIN + N_TEST, seed=seed)
    tr, te = np.arange(N_TRAIN), np.arange(N_TRAIN, N_TRAIN + N_TEST)
    coords = ds.coords[tr]
    y_true = ds.y[tr]
    y_noisy, is_noisy = inject_noise(y_true, coords, noise_type, rate, seed=seed + 1000)
    n_noisy = int(is_noisy.sum())
    base = dict(noise=noise_type, rate=rate, seed=seed)

    det_rows, down_rows, probas = [], [], {}
    nbr = d.neighbour_disagreement(coords, y_noisy, K)
    det_rows.append({**base, "features": "-", "cv": "-", "detector": "neighbour_vote", **detection_metrics(is_noisy, nbr)})

    rankings = {"neighbour_vote|-|-": nbr}
    for feature_set, cv in itertools.product(("spectral", "spectral+xy"), ("random", "spatial")):
        X = features(ds, tr, feature_set)
        p = d.out_of_fold_proba(X, y_noisy, coords, cv=cv, block_size=BLOCK_SIZE, model=rf(seed), n_classes=6, seed=seed)
        probas[(feature_set, cv)] = p
        ps = d.smooth_proba(p, coords, K)
        for name, proba in (("self_confidence", p), ("smoothed_sc", ps)):
            score = d.self_confidence(proba, y_noisy)
            flags = flag_metrics(is_noisy, d.confident_learning_flags(proba, y_noisy))
            det_rows.append({
                **base, "features": feature_set, "cv": cv, "detector": name,
                **detection_metrics(is_noisy, score),
                **{f"cl_{k}": v for k, v in flags.items()},
            })
            rankings[f"{name}|{feature_set}|{cv}"] = score

    # Downstream: drop the top-n suspicious samples (n = true number of errors), retrain, test on clean labels.
    if n_noisy:
        def evaluate(method, keep, labels, feature_set):
            m = rf(seed).fit(features(ds, tr[keep], feature_set), labels[keep])
            pred = m.predict(features(ds, te, feature_set))
            down_rows.append({**base, "features": feature_set, "method": method,
                              "accuracy": accuracy_score(ds.y[te], pred),
                              "macro_f1": f1_score(ds.y[te], pred, average="macro")})

        all_idx = np.ones(N_TRAIN, dtype=bool)
        for fs in ("spectral", "spectral+xy"):
            evaluate("no_cleaning", all_idx, y_noisy, fs)
            evaluate("oracle_clean_labels", all_idx, y_true, fs)
            evaluate("oracle_remove_noisy", ~is_noisy, y_noisy, fs)
            for key, score in rankings.items():
                name, rfs, cv = key.split("|")
                if rfs not in ("-", fs):
                    continue
                keep = np.ones(N_TRAIN, dtype=bool)
                keep[np.argsort(-score, kind="stable")[:n_noisy]] = False
                evaluate(f"remove:{name}|{cv}", keep, y_noisy, fs)

    return det_rows, down_rows, probas, (coords, y_noisy, is_noisy, ds, tr)


def ablation_smoothing(probas, coords, y_noisy, is_noisy, base):
    rows = []
    for (feature_set, cv), p in probas.items():
        for k in (5, 10, 20, 40):
            for w in (0.0, 0.25, 0.5, 0.75, 1.0):
                score = d.self_confidence(d.smooth_proba(p, coords, k, w), y_noisy)
                rows.append({**base, "features": feature_set, "cv": cv, "k": k, "self_weight": w,
                             **detection_metrics(is_noisy, score)})
    return rows


def ablation_block_size(ds, tr, coords, y_noisy, is_noisy, base, seed):
    rows = []
    X = features(ds, tr, "spectral+xy")
    for bs in (2.5, 5.0, 10.0, 20.0):
        p = d.out_of_fold_proba(X, y_noisy, coords, cv="spatial", block_size=bs, model=rf(seed), n_classes=6, seed=seed)
        for name, proba in (("self_confidence", p), ("smoothed_sc", d.smooth_proba(p, coords, K))):
            rows.append({**base, "block_size": bs, "detector": name,
                         **detection_metrics(is_noisy, d.self_confidence(proba, y_noisy))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()
    seeds = range(1 if args.quick else args.seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Results are saved after every config; a re-run skips configs already on disk.
    names = ("detection", "downstream", "ablation_smoothing", "ablation_block_size")
    tables = {n: pd.read_csv(out / f"{n}.csv") if (out / f"{n}.csv").exists() else pd.DataFrame() for n in names}
    done = set()
    if not tables["detection"].empty:
        done = set(map(tuple, tables["detection"][["noise", "rate", "seed"]].drop_duplicates().to_numpy().tolist()))

    configs = list(itertools.product(NOISE_TYPES, RATES, seeds))
    t0 = time.time()
    for i, (nt, rate, seed) in enumerate(configs, 1):
        if (nt, rate, seed) in done:
            continue
        rows, drows, probas, (coords, y_noisy, is_noisy, ds, tr) = run_config(nt, rate, seed)
        new = {"detection": rows, "downstream": drows, "ablation_smoothing": [], "ablation_block_size": []}
        base = dict(noise=nt, rate=rate, seed=seed)
        if rate == 0.2:
            new["ablation_smoothing"] = ablation_smoothing(probas, coords, y_noisy, is_noisy, base)
            if nt == "clustered":
                new["ablation_block_size"] = ablation_block_size(ds, tr, coords, y_noisy, is_noisy, base, seed)
        for n in names:
            if new[n]:
                tables[n] = pd.concat([tables[n], pd.DataFrame(new[n])], ignore_index=True)
                tables[n].to_csv(out / f"{n}.csv", index=False)
        print(f"[{i}/{len(configs)}] {nt:9s} rate={rate} seed={seed}  {time.time() - t0:6.0f}s", flush=True)

    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
