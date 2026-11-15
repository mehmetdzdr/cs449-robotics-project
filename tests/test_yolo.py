import time

import numpy as np
import pytest

from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.dataset import CLASSES, randomize_scene
from conveyor_sort.perception.yolo import DEFAULT_WEIGHTS, YoloDetector, boxes_to_detections
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World


def test_ground_truth_boxes_are_localized_and_boxes_under_an_arm_are_dropped():
    """The part that needs no model: bounding box + depth -> Detection."""
    world = World(ConveyorConfig())
    camera, rng = Camera(world.C), np.random.default_rng(0)
    errors = []
    for _ in range(10):
        randomize_scene(world, rng)
        obs = camera.capture(timestamp=2.0)
        names = [n for n in world.conveyor.on_belt() if obs.frame_mask(world.C, n).sum() > 600]   # well visible
        boxes = []
        for name in names:
            vs, us = np.nonzero(obs.frame_mask(world.C, name))
            boxes.append([us.min(), vs.min(), us.max() + 1, vs.max() + 1])
        classes = [CLASSES.index(world.conveyor.colors[n].value) for n in names]
        detections = boxes_to_detections(np.array(boxes), np.array(classes), np.full(len(names), 0.9), obs,
                                         world.conveyor.geometry)
        assert len(detections) == len(names)
        for name, d in zip(names, detections):
            assert d.color == world.conveyor.colors[name] and d.confidence == 0.9 and d.timestamp == 2.0
            errors.append(np.linalg.norm(d.position - world.C.getFrame(name).getPosition()))
    assert len(errors) > 30 and np.mean(errors) < 0.003 and np.max(errors) < 0.006
    print(f"{len(errors)} boxes, mean error {np.mean(errors) * 1000:.1f} mm, max {np.max(errors) * 1000:.1f} mm")

    table = np.array([[100, 600, 130, 630]])                 # a rectangle on the table, not at box height
    assert boxes_to_detections(table, np.array([0]), np.array([0.9]), obs, world.conveyor.geometry) == []


@pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="fine-tuned weights are not in git (see README)")
def test_yolo_detector_against_ground_truth_on_random_scenes():
    pytest.importorskip("ultralytics")
    world = World(ConveyorConfig(seed=5))
    camera, detector = Camera(world.C), YoloDetector(world.conveyor.geometry)
    rng = np.random.default_rng(123)                         # scenes that are not in the training set (seed 0)
    errors, false_positives, seconds = [], 0, []
    detector.detect(camera.capture())                        # warm-up, not timed
    for _ in range(25):
        randomize_scene(world, rng)
        obs = camera.capture(timestamp=1.5)
        start = time.perf_counter()
        detections = detector.detect(obs)
        seconds.append(time.perf_counter() - start)
        boxes = {n: world.C.getFrame(n).getPosition() for n in world.conveyor.on_belt()}
        for name, position in boxes.items():
            if obs.frame_mask(world.C, name).sum() < 600:      # box largely hidden by an arm: not required
                continue
            distances = [np.linalg.norm(d.position - position) for d in detections
                         if d.color == world.conveyor.colors[name]]
            assert distances and min(distances) < 0.03, f"{name} not detected"
            errors.append(min(distances))
        for d in detections:                                    # every detection must lie on a box of its color
            if not any(world.conveyor.colors[n] == d.color and np.linalg.norm(d.position - p) < 0.04
                       for n, p in boxes.items()):
                false_positives += 1
    assert len(errors) > 80
    assert false_positives == 0
    assert np.mean(errors) < 0.003
    print(f"{len(errors)} boxes, mean error {np.mean(errors) * 1000:.1f} mm, max {np.max(errors) * 1000:.1f} mm, "
          f"detect() {np.mean(seconds) * 1000:.0f} ms per frame")
