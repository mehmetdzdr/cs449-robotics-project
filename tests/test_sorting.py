import numpy as np

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.hsv import HsvDetector
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.scene import HOME_Q
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots, run_single_robot

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


def test_two_robots_with_fixed_roles_work_at_the_same_time():
    config = ConveyorConfig(speed=0.1, spacing=0.6, seed=0)              # a box every 6 s, 1 in 4 gray
    world = World(config)
    both_moving, collisions = [0], []

    def check(world):
        q = world.C.getJointState()
        both_moving[0] += bool(np.abs(q[:7] - HOME_Q).max() > 0.05 and np.abs(q[7:] - HOME_Q).max() > 0.05)
        world.C.computeCollisions()
        collisions.extend(world.C.getCollisions(0.0))

    stats = run_robots(world, GroundTruthDetector(world).detect, duration=60.0, on_step=check)
    by_robot = {r: [e for e in stats.events if e.robot_id == r] for r in (1, 2)}
    assert len(by_robot[1]) == 5 and len(by_robot[2]) == 2
    assert all(e.color != BoxColor.GRAY and e.region != "bin_gray" for e in by_robot[1])
    assert all(e.color == BoxColor.GRAY and e.region == "bin_gray" for e in by_robot[2])
    assert stats.sorted_correctly == 7 and stats.failed == 0 and stats.missed == 0
    assert both_moving[0] * world.dt > 10.0                              # the arms really move simultaneously
    assert collisions == []

    single = World(config)                                               # same boxes, Robot 1 alone: the gray ones leave the belt
    assert run_single_robot(single, GroundTruthDetector(single).detect, duration=60.0).missed == 3
