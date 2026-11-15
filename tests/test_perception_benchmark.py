import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor, Detection
from conveyor_sort.perception.benchmark import TOUCH_GAP, evaluate, make_scenes
from conveyor_sort.perception.hsv import HsvDetector


@pytest.fixture(scope="module")
def scenes():
    return make_scenes(6, seed=7)


def test_hsv_on_random_scenes(scenes):
    hsv = HsvDetector(scenes[0][0].conveyor.geometry)
    m = evaluate(scenes, hsv.detect, lambda obs: [mask for _, mask, _ in hsv.masks(obs)])
    assert m["boxes"] > 15 and m["recall"] == 1.0 and m["precision"] == 1.0 and m["wrong_color"] == 0.0
    assert m["error_mean_mm"] < 3.0 and m["mask_iou"] > 0.9 and 0 < m["time_ms"] < 500


def test_metrics_count_misses_wrong_colors_and_false_positives(scenes):
    def truth(obs, keep=lambda i: True, recolor=False, extra=False):
        world = next(w for w, o in scenes if o is obs)
        names = [n for n in world.conveyor.on_belt() if obs.frame_mask(world.C, n).sum() >= 600]
        detections = [Detection(BoxColor.RED if recolor else world.conveyor.colors[n], world.C.getFrame(n).getPosition(), 1.0, 0.0)
                      for i, n in enumerate(names) if keep(i)]
        return detections + ([Detection(BoxColor.RED, np.array([0.0, -0.5, 0.65]), 1.0, 0.0)] if extra else [])

    perfect = evaluate(scenes, truth)
    assert perfect["recall"] == 1.0 and perfect["precision"] == 1.0 and perfect["error_max_mm"] == 0.0
    assert np.isnan(perfect["mask_iou"])                                        # no masks given
    half = evaluate(scenes, lambda obs: truth(obs, keep=lambda i: i % 2 == 0))
    assert 0.4 < half["recall"] < 0.7 and half["precision"] == 1.0
    red = evaluate(scenes, lambda obs: truth(obs, recolor=True))
    assert red["recall"] + red["wrong_color"] == pytest.approx(1.0) and red["wrong_color"] > 0.5
    extra = evaluate(scenes, lambda obs: truth(obs, extra=True))
    assert extra["recall"] == 1.0 and extra["precision"] < 1.0


def test_dense_arrangement_has_touching_boxes_and_brightness_scales_the_image():
    (world, obs), = make_scenes(1, seed=3, arrangement="dense")
    positions = [world.C.getFrame(n).getPosition() for n in world.conveyor.on_belt()]
    gaps = [np.linalg.norm(a - b) for i, a in enumerate(positions) for b in positions[i + 1:]]
    assert min(gaps) == pytest.approx(TOUCH_GAP, abs=1e-6)
    (_, dark), = make_scenes(1, seed=3, brightness=0.5)
    (_, normal), = make_scenes(1, seed=3)
    assert dark.rgb.mean() == pytest.approx(normal.rgb.mean() / 2, rel=0.02)
