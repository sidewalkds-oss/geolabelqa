import numpy as np
import pytest
from scipy.io import savemat

from geolabelqa.scenes import load_scene, most_confused_partner
from geolabelqa.synthetic import inject_noise


@pytest.fixture
def fake_scene(tmp_path):
    rng = np.random.default_rng(0)
    gt = np.zeros((40, 50), dtype=np.uint8)
    gt[:20, :25], gt[20:, :25], gt[:, 25:] = 1, 2, 3
    gt[0, 0] = 9  # a one-pixel class, dropped by min_class_size
    cube = rng.integers(0, 5000, size=(40, 50, 6)).astype(np.uint16)
    savemat(tmp_path / "Indian_pines_corrected.mat", {"indian_pines_corrected": cube})
    savemat(tmp_path / "Indian_pines_gt.mat", {"indian_pines_gt": gt})
    return tmp_path, gt


def test_load_scene(fake_scene):
    path, gt = fake_scene
    scene = load_scene("indian_pines", path, min_class_size=100)
    assert scene.class_ids.tolist() == [1, 2, 3]
    assert len(scene.y) == int((gt > 0).sum()) - 1
    assert scene.X.shape == (len(scene.y), 6) and scene.X.max() == pytest.approx(1.0)
    assert scene.coords[:, 0].max() == pytest.approx(49 * 20.0)
    assert scene.extent == pytest.approx(50 * 20.0)


def test_load_scene_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="ehu.eus"):
        load_scene("pavia_university", tmp_path)


def test_most_confused_partner():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.1, 0.4, 0.5]])
    y = np.array([0, 1, 2])
    assert most_confused_partner(proba, y).tolist() == [1, 2, 1]


def test_inject_noise_custom_partner():
    y = np.zeros(100, dtype=int)
    noisy, flip = inject_noise(y, np.zeros((100, 2)), "pair", 0.3, n_classes=3, seed=0, partner=np.array([2, 0, 1]))
    assert (noisy[flip] == 2).all()
