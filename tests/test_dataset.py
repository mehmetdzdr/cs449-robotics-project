import numpy as np

from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.dataset import CLASSES, generate, randomize_scene, yolo_labels
from conveyor_sort.sim.conveyor import BOX_RGB, ConveyorConfig
from conveyor_sort.sim.world import World


def test_labels_match_rendered_boxes():
    """Every label must sit on a box of the labelled color, also after the scene was re-randomized."""
    world = World(ConveyorConfig())
    camera = Camera(world.C)
    rng = np.random.default_rng(0)
    n_labels = 0
    for _ in range(6):
        randomize_scene(world, rng)
        obs = camera.capture()
        height, width = obs.depth.shape
        for line in yolo_labels(world, obs):
            cls, xc, yc, w, h = line.split()
            assert 0.015 < float(w) < 0.04 and 0.03 < float(h) < 0.06          # about 30 x 30 px
            rgb = obs.rgb[round(float(yc) * height), round(float(xc) * width)] / 255
            expected = np.array(BOX_RGB[next(c for c in BOX_RGB if c.value == CLASSES[int(cls)])])
            assert np.abs(rgb - expected).max() < 0.35, (line, rgb)
            n_labels += 1
    assert n_labels > 15


def test_generate_writes_yolo_layout(tmp_path):
    generate(tmp_path, n_images=5, val_fraction=0.2, seed=1)
    assert len(list((tmp_path / "images" / "train").glob("*.png"))) == 4
    assert len(list((tmp_path / "images" / "val").glob("*.png"))) == 1
    for image in tmp_path.glob("images/*/*.png"):
        assert (tmp_path / "labels" / image.parent.name / f"{image.stem}.txt").exists()
    yaml = (tmp_path / "data.yaml").read_text()
    assert "train: images/train" in yaml and "3: gray" in yaml
