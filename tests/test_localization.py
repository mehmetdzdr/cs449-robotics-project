import cv2
import numpy as np
import pytest

from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.localization import mask_to_world, pixels_to_world
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World


@pytest.fixture(scope="module")
def scene():
    """Belt filled end to end with boxes, so the image centre and both image edges are covered."""
    world = World(ConveyorConfig(speed=0.2, spacing=0.15))
    world.run(10.5)
    return world, Camera(world.C).capture(world.time)


def errors_mm(world, obs, transform=lambda mask: mask):
    names = world.conveyor.on_belt()
    assert len(names) >= 14
    return np.array([
        np.linalg.norm(mask_to_world(transform(obs.frame_mask(world.C, n)), obs) - world.C.getFrame(n).getPosition())
        for n in names
    ]) * 1000


def test_localization_error_with_ground_truth_masks(scene):
    errors = errors_mm(*scene)
    assert errors.mean() < 3.0
    assert errors.max() < 5.0


def test_localization_tolerates_oversized_masks(scene):
    """A detector's mask may spill onto the belt; the top-face filter must ignore those pixels."""
    kernel = np.ones((7, 7), np.uint8)
    errors = errors_mm(*scene, transform=lambda m: cv2.dilate(m.astype(np.uint8), kernel).astype(bool))
    assert errors.max() < 5.0


def test_empty_mask_returns_none(scene):
    _, obs = scene
    assert mask_to_world(np.zeros(obs.depth.shape, bool), obs) is None


def test_image_centre_projects_below_camera(scene):
    _, obs = scene
    point = pixels_to_world(np.array([640.0]), np.array([360.0]), np.array([2.0]), obs)[0]
    assert point == pytest.approx(obs.cam_position - [0, 0, 2.0])
