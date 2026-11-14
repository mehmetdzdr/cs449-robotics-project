import numpy as np

from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.dataset import randomize_scene
from conveyor_sort.perception.hsv import HsvDetector, belt_roi
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World


def test_hsv_detector_against_ground_truth_on_random_scenes():
    """Random box arrangements with the arms partly over the belt (source of gray false positives)."""
    world = World(ConveyorConfig())
    camera, detector = Camera(world.C), HsvDetector(world.conveyor.geometry)
    rng = np.random.default_rng(0)
    errors, false_positives = [], 0
    for _ in range(25):
        randomize_scene(world, rng)
        obs = camera.capture(timestamp=1.5)
        detections = detector.detect(obs)
        boxes = {n: world.C.getFrame(n).getPosition() for n in world.conveyor.on_belt()}
        for name, position in boxes.items():
            if obs.frame_mask(world.C, name).sum() < 600:      # box largely hidden by an arm: not required
                continue
            distances = [np.linalg.norm(d.position - position) for d in detections
                         if d.color == world.conveyor.colors[name]]
            assert distances and min(distances) < 0.03, f"{name} not detected"
            errors.append(min(distances))
        for d in detections:                                    # every detection must lie on a box of its color
            assert d.timestamp == 1.5 and d.confidence == 1.0
            if not any(world.conveyor.colors[n] == d.color and np.linalg.norm(d.position - p) < 0.04
                       for n, p in boxes.items()):
                false_positives += 1
    assert len(errors) > 80
    assert false_positives == 0
    assert np.mean(errors) < 0.003
    print(f"{len(errors)} boxes, mean error {np.mean(errors) * 1000:.1f} mm, max {np.max(errors) * 1000:.1f} mm")


def test_belt_roi_covers_the_belt_only():
    world = World(ConveyorConfig())
    obs = Camera(world.C).capture()
    rows, cols = belt_roi(obs, world.conveyor.geometry)
    belt_mask = obs.frame_mask(world.C, "belt")
    assert belt_mask[rows, cols].sum() == belt_mask.sum()                      # whole belt inside
    assert (rows.stop - rows.start) * (cols.stop - cols.start) < 1.4 * belt_mask.sum()   # and little else
