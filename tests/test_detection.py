import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from geolabelqa import LabelAuditor, SpatialBlockKFold
from geolabelqa import detectors as d
from geolabelqa.metrics import detection_metrics, flag_metrics
from geolabelqa.synthetic import CLASSES, inject_noise, make_landcover


@pytest.fixture(scope="module")
def ds():
    return make_landcover(n_samples=800, seed=3)


def small_rf():
    return RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1)


def test_make_landcover_shapes(ds):
    assert ds.X.shape == (800, 9)
    assert ds.coords.shape == (800, 2)
    assert set(np.unique(ds.y)) <= set(range(len(CLASSES)))
    assert (ds.coords >= 0).all() and (ds.coords <= ds.extent).all()


def test_landcover_is_spatially_autocorrelated(ds):
    # Near neighbours should share a class far more often than chance.
    agreement = 1 - d.neighbour_disagreement(ds.coords, ds.y, k=5).mean()
    chance = (np.bincount(ds.y) / len(ds.y)) @ (np.bincount(ds.y) / len(ds.y))
    assert agreement > chance + 0.3


@pytest.mark.parametrize("noise_type", ["uniform", "pair", "clustered"])
def test_inject_noise_rate_and_flips(ds, noise_type):
    noisy, flip = inject_noise(ds.y, ds.coords, noise_type, 0.2, seed=0)
    assert flip.sum() == round(0.2 * len(ds.y))
    assert (noisy[flip] != ds.y[flip]).all()
    assert (noisy[~flip] == ds.y[~flip]).all()


def test_clustered_noise_is_clustered(ds):
    _, flip_c = inject_noise(ds.y, ds.coords, "clustered", 0.2, seed=0)
    _, flip_u = inject_noise(ds.y, ds.coords, "uniform", 0.2, seed=0)
    # Clustered errors have many more erroneous neighbours.
    nn_c = 1 - d.neighbour_disagreement(ds.coords, flip_c.astype(int), k=5)[flip_c].mean()
    nn_u = 1 - d.neighbour_disagreement(ds.coords, flip_u.astype(int), k=5)[flip_u].mean()
    assert nn_c > nn_u + 0.3


def test_inject_noise_validates():
    with pytest.raises(ValueError):
        inject_noise(np.zeros(5, int), np.zeros((5, 2)), "bogus", 0.1)


def test_spatial_block_kfold_partitions_blocks(ds):
    cv = SpatialBlockKFold(n_splits=5, block_size=10.0, seed=0)
    blocks = cv.block_ids(ds.coords)
    seen = np.zeros(len(ds.y), int)
    for train, test in cv.split(ds.coords):
        seen[test] += 1
        assert not set(blocks[train]) & set(blocks[test])
    assert (seen == 1).all()


def test_spatial_block_kfold_buffer(ds):
    cv = SpatialBlockKFold(n_splits=5, block_size=10.0, buffer=2.0, seed=0)
    for train, test in cv.split(ds.coords):
        gaps = np.min(np.hypot(*(ds.coords[train][:, None] - ds.coords[test][None]).transpose(2, 0, 1)), axis=1)
        assert gaps.min() > 2.0


def test_spatial_block_kfold_too_few_blocks(ds):
    with pytest.raises(ValueError):
        list(SpatialBlockKFold(n_splits=5, block_size=1000.0).split(ds.coords))


def test_out_of_fold_proba_rows_sum_to_one(ds):
    for cv in ("random", "spatial"):
        p = d.out_of_fold_proba(ds.X, ds.y, ds.coords, cv=cv, model=small_rf(), n_classes=6)
        assert p.shape == (len(ds.y), 6)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_confident_learning_flags_obvious_error():
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.95, 0.05]])
    y = np.array([0, 0, 1, 1, 1])  # last label contradicts a confident prediction
    assert d.confident_learning_flags(proba, y).tolist() == [False, False, False, False, True]


def test_rank_average_bounds():
    s = d.rank_average(np.array([1.0, 2, 3]), np.array([30.0, 20, 10]))
    np.testing.assert_allclose(s, [2 / 3, 2 / 3, 2 / 3])


def test_detectors_beat_chance(ds):
    noisy, flip = inject_noise(ds.y, ds.coords, "uniform", 0.2, seed=1)
    p = d.out_of_fold_proba(ds.X, noisy, ds.coords, cv="spatial", model=small_rf(), n_classes=6)
    for score in (d.self_confidence(p, noisy), d.smoothed_self_confidence(p, noisy, ds.coords)):
        assert detection_metrics(flip, score)["auroc"] > 0.8


def test_metrics_edge_cases():
    m = detection_metrics(np.zeros(4, bool), np.arange(4.0))
    assert np.isnan(m["auroc"])
    f = flag_metrics(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    assert f["precision"] == 0.5 and f["recall"] == 0.5 and f["f1"] == 0.5


def test_label_auditor_ranks_errors_first(ds):
    noisy, flip = inject_noise(ds.y, ds.coords, "clustered", 0.15, seed=2)
    labels = np.array(CLASSES)[noisy]
    report = LabelAuditor(model=small_rf()).audit(ds.X, labels, ds.coords)
    assert list(report.columns) == ["index", "label", "suggested_label", "suspicion", "cl_flag", "neighbour_disagreement"]
    assert report["suspicion"].is_monotonic_decreasing
    top = report["index"].to_numpy()[: flip.sum()]
    assert flip[top].mean() > 3 * flip.mean()
