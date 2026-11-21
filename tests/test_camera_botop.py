"""Camera images from BotOp (#94): rgb and depth from the simulated camera, segmentation from the synced C."""
import time

import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.detectors import get_detector
from conveyor_sort.perception.hsv import HsvDetector
from conveyor_sort.perception.localization import mask_to_world
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World

COLORS = [BoxColor.RED, BoxColor.GREEN, BoxColor.BLUE, BoxColor.GRAY]


@pytest.fixture
def physics_world():
    worlds = []

    def make(speed: float) -> tuple[World, list[str]]:
        """Eight boxes along the belt at random lateral offsets, settled."""
        world = World(ConveyorConfig(speed=speed, max_boxes=0), physics=True)
        worlds.append(world)
        rng = np.random.default_rng(0)
        boxes = [world.conveyor.spawn_box(COLORS[i % 4], x=-0.9 + 0.22 * i, y=rng.uniform(-0.08, 0.08))
                 for i in range(8)]
        world.run(1.0)
        return world, boxes

    yield make
    for world in worlds:
        world.close()


def test_image_and_depth_match_the_camera_view(physics_world):
    world, _ = physics_world(0.0)
    world.run(1.0)                                   # a new BotOp image of the settled state
    botop = Camera(world.C, bot=world.bot).capture(world.time)
    view = Camera(world.C).capture(world.time)
    rgb_diff = np.abs(botop.rgb.astype(int) - view.rgb.astype(int)).max(axis=2)
    valid = (botop.depth > 0) & (view.depth > 0)
    depth_diff = np.abs(botop.depth - view.depth)[valid]
    print(f"rgb: {(rgb_diff > 10).mean() * 100:.4f} % of the pixels differ by more than 10; depth: mean "
          f"{depth_diff.mean() * 1000:.4f} mm, {(depth_diff > 0.001).mean() * 100:.4f} % of the pixels off by more "
          f"than 1 mm (max {depth_diff.max() * 1000:.0f} mm: silhouette edges)")
    assert botop.rgb.shape == view.rgb.shape and botop.depth.shape == view.depth.shape
    assert np.allclose(botop.fxycxy, view.fxycxy)
    assert (rgb_diff > 10).mean() < 0.001 and (depth_diff > 0.001).mean() < 0.001
    assert np.array_equal(botop.segmentation, view.segmentation)


def test_localization_with_ground_truth_masks_on_physical_boxes(physics_world):
    world, boxes = physics_world(0.0)          # stopped: the segmentation (current state) matches the image
    camera = Camera(world.C, bot=world.bot)
    errors = []
    for _ in range(3):
        world.run(0.5)
        obs = camera.capture(world.time)
        for box in boxes:
            estimate = mask_to_world(obs.frame_mask(world.C, box), obs)
            errors.append(np.linalg.norm(estimate - world.C.getFrame(box).getPosition()))
    errors = np.array(errors)
    print(f"localization error {errors.mean() * 1000:.1f} mm mean, {errors.max() * 1000:.1f} mm max "
          f"({len(errors)} box views)")
    assert errors.mean() < 0.003 and errors.max() < 0.005


@pytest.mark.parametrize("speed", [0.1, 0.2])
def test_image_timestamps_on_the_moving_belt(physics_world, speed):
    """HSV detections are compared with where the boxes were at the observation's timestamp."""
    world, boxes = physics_world(speed)
    detect = get_detector("hsv", world)
    errors, new_images, steps, t0 = [], 0, 0, world.time
    while steps < 60:                                  # 60 steps (each longer than dt: capture renders)
        world.step()
        steps += 1
        detections = detect()
        new_images += bool(detections)
        for d in detections:      # not at the belt end: those boxes are taken out, the image may still show them
            if d.position[0] > world.conveyor.geometry.x_end - 0.1:
                continue
            truth = {b: world.C.getFrame(b).getPosition() - [speed * (world.time - d.timestamp), 0, 0] for b in boxes}
            errors.append(min(np.linalg.norm(p - d.position) for p in truth.values()))
    errors = np.array(errors)
    print(f"{speed} m/s: {new_images} new images in {world.time - t0:.1f} s, error at the image time {errors.mean() * 1000:.1f} mm mean, "
          f"{errors.max() * 1000:.1f} mm max")
    # a new image every 0.35 to 0.5 s here, 0.64 s on the CI runner (software rendering on 2 cores); repeats are not
    # reported (they would bring the interval down to the step, 0.05 s)
    assert 0.3 <= (world.time - t0) / new_images <= 0.8
    # the image time is only estimated (see Camera)
    assert errors.mean() < 0.004 and errors.max() < 0.015


def test_hsv_detector_timing_is_unchanged(physics_world):
    world, boxes = physics_world(0.0)
    detector = HsvDetector(world.conveyor.geometry)
    timings = {}
    for name, camera in (("botop", Camera(world.C, bot=world.bot)), ("view", Camera(world.C))):
        obs = camera.capture(world.time)
        detector.detect(obs)                          # warm-up
        start = time.perf_counter()
        detections = detector.detect(obs)
        timings[name] = time.perf_counter() - start
        print(f"{name}: {len(detections)} detections, detect {timings[name] * 1000:.1f} ms")
        assert len(detections) == len(boxes)
    assert timings["botop"] < 1.5 * timings["view"] + 0.005


def test_closing_the_world_releases_the_simulation_from_cameras_and_detectors():
    """A detector (its camera) or ArmReferences that outlive the world must not keep BotOp alive: its destructor then
    ran later in garbage collection, while other simulations ran, and crashed in PhysX (CI segfaults since #154)."""
    world = World(ConveyorConfig(max_boxes=0), physics=True)
    camera = Camera(world.C, bot=world.bot)
    detect, arms = get_detector("hsv", world), world.arms
    detect()
    world.close()
    assert camera.bot is None and arms.bot is None and world.bot is None
    assert camera.capture(world.time).rgb.shape == (720, 1280, 3)     # it still renders, from C
    assert detect() == []                                              # no boxes on the belt
