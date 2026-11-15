import time

import numpy as np
import pytest

from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.dataset import randomize_scene
from conveyor_sort.perception.yolo import DEFAULT_WEIGHTS, localize
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World


def test_localize_uses_only_the_box_top_of_a_mask():
    """The part that needs no model: mask + depth -> position, with ground-truth masks that are too generous."""
    world = World(ConveyorConfig())
    camera, rng = Camera(world.C), np.random.default_rng(1)
    errors = []
    for _ in range(8):
        randomize_scene(world, rng)
        obs = camera.capture()
        for name in world.conveyor.on_belt():
            mask = obs.frame_mask(world.C, name)
            if mask.sum() < 600:
                continue
            vs, us = np.nonzero(mask)
            grown = np.zeros_like(mask)
            grown[vs.min() - 4:vs.max() + 5, us.min() - 4:us.max() + 5] = True       # also belt and maybe an arm
            errors.append(np.linalg.norm(localize(grown, obs, world.conveyor.geometry) - world.C.getFrame(name).getPosition()))
    assert len(errors) > 25 and np.mean(errors) < 0.003 and np.max(errors) < 0.01     # worst: a box partly under an arm
    table = np.zeros(obs.depth.shape, bool)
    table[600:630, 100:130] = True
    assert localize(table, obs, world.conveyor.geometry) is None                       # nothing at box height there


@pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="fine-tuned YOLO weights are not in git (see README)")
@pytest.mark.parametrize("model", ["mobile_sam", "sam_b"])
def test_sam_detector_against_ground_truth_on_random_scenes(model):
    pytest.importorskip("ultralytics")
    from conveyor_sort.perception.sam import SamDetector

    world = World(ConveyorConfig(seed=5))
    camera, detector = Camera(world.C), SamDetector(world.conveyor.geometry, model)
    rng = np.random.default_rng(123)
    errors, ious, false_positives, seconds = [], [], 0, []
    randomize_scene(world, rng)
    detector.detect(camera.capture())                        # warm-up, not timed
    for _ in range(15):
        randomize_scene(world, rng)
        obs = camera.capture(timestamp=1.5)
        start = time.perf_counter()
        detections = detector.detect(obs)
        seconds.append(time.perf_counter() - start)
        masks = detector.masks(obs)[0]
        boxes = {n: world.C.getFrame(n).getPosition() for n in world.conveyor.on_belt()}
        for name, position in boxes.items():
            truth = obs.frame_mask(world.C, name)
            if truth.sum() < 600:                              # box largely hidden by an arm: not required
                continue
            distances = [np.linalg.norm(d.position - position) for d in detections
                         if d.color == world.conveyor.colors[name]]
            assert distances and min(distances) < 0.03, f"{name} not detected"
            errors.append(min(distances))
            ious.append(max((m & truth).sum() / (m | truth).sum() for m in masks))
        for d in detections:
            if not any(world.conveyor.colors[n] == d.color and np.linalg.norm(d.position - p) < 0.04
                       for n, p in boxes.items()):
                false_positives += 1
    assert len(errors) > 40 and false_positives == 0
    assert np.mean(errors) < 0.003 and np.mean(ious) > 0.9
    print(f"{model}: {len(errors)} boxes, mean error {np.mean(errors) * 1000:.1f} mm, max {np.max(errors) * 1000:.1f} mm, "
          f"mask IoU {np.mean(ious):.3f} (min {np.min(ious):.3f}), detect() {np.mean(seconds) * 1000:.0f} ms per frame")
