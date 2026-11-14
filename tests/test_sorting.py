import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.hsv import HsvDetector
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_single_robot

COLORED = {BoxColor.RED: 1.0, BoxColor.GREEN: 1.0, BoxColor.BLUE: 1.0}


def sparse_world(seed=0):
    """One colored box every 12 s: slower than the robot's pick-and-place cycle, so none should be missed."""
    return World(ConveyorConfig(speed=0.1, spacing=1.2, color_weights=COLORED, seed=seed))


def test_single_robot_sorts_every_box_with_ground_truth():
    world = sparse_world()
    stats = run_single_robot(world, GroundTruthDetector(world).detect, duration=60.0)
    assert len(stats.events) == 5                      # boxes entered at t = 0, 12, 24, 36, 48 s
    assert stats.sorted_correctly == 5
    assert stats.failed == 0 and stats.missed == 0
    assert {e.region for e in stats.events} <= {"region_A", "region_B", "region_C"}


def test_single_robot_sorts_with_hsv_vision():
    world = sparse_world(seed=1)
    camera, detector = Camera(world.C), HsvDetector(world.conveyor.geometry)
    stats = run_single_robot(world, lambda: detector.detect(camera.capture(world.time)), duration=13.0)   # one box: rendering every step is slow
    assert len(stats.events) == 1
    assert stats.sorted_correctly == 1
    assert stats.failed == 0 and stats.missed == 0


def test_gray_boxes_are_left_on_the_belt_and_dense_traffic_causes_misses():
    world = World(ConveyorConfig(speed=0.2, spacing=0.3, seed=2))        # a box every 1.5 s, 1 in 4 gray
    stats = run_single_robot(world, GroundTruthDetector(world).detect, duration=40.0)
    assert stats.sorted_correctly >= 2
    assert stats.sorted_correctly == len(stats.events) - stats.failed
    assert all(e.color != BoxColor.GRAY for e in stats.events)
    assert stats.missed > 5                             # one robot cannot keep up
