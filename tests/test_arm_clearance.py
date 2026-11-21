"""Arm motions checked against the other arm's motion (#141), physics backend."""
import numpy as np
import pytest

from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.metrics import MODES
from conveyor_sort.physics.contacts import arm_proximity
from conveyor_sort.planning.execute import follow_path
from conveyor_sort.planning.pick import plan_pick
from conveyor_sort.planning.pick_place import CLEARANCE, PickPlaceTask
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots

pytestmark = pytest.mark.slow   # physics scenarios in real time: nightly / on demand in CI (#97)


@pytest.mark.parametrize("x2, close", [(0.05, True), (0.45, False)])
def test_clearance_of_a_path_against_the_moving_other_arm(x2, close):
    """Robot 2 starts towards a box at x2; Robot 1's path to a box at -0.05 is checked against that motion."""
    world = World(ConveyorConfig(speed=0.0, max_boxes=0), physics=True)
    try:
        box1 = world.conveyor.spawn_box(BoxColor.RED, x=-0.05, y=0.0)
        box2 = world.conveyor.spawn_box(BoxColor.RED, x=x2, y=0.0)
        world.run(0.5)
        path1, path2 = plan_pick(world.C, 1, box1).path, plan_pick(world.C, 2, box2).path
        next(follow_path(world, path2, 2.0, robot_id=2))           # Robot 2 is on its way (2 s)
        clearance = world.arms.clearance(world.C, 1, path1, 2.0)
        print(f"Robot 2 to x = {x2}: Robot 1's path {clearance * 100:.1f} cm from it")
        assert (clearance < CLEARANCE) == close
    finally:
        world.close()


@pytest.mark.parametrize("name", ["coordinated/v0.05/n10/s0.3/green-heavy", "fixed/v0.15/n20/s0.3/uniform"])
def test_streams_that_had_arm_contacts(name):
    """The two streams with the most arm contacts in the benchmark of #107 (16 and 11 contacts in single runs)."""
    mode, v, n, s, colors = name.split("/")
    weights = {"uniform": {c: 1.0 for c in BoxColor},
               "green-heavy": {BoxColor.RED: 1, BoxColor.GREEN: 3, BoxColor.BLUE: 1, BoxColor.GRAY: 1}}[colors]
    speed, boxes, spacing = float(v[1:]), int(n[1:]), float(s[1:])
    world = World(ConveyorConfig(speed, spacing, color_weights=weights, seed=0, max_boxes=boxes), physics=True)
    try:
        roles, clear_and_sort = MODES[mode]
        duration = ((boxes - 1) * spacing + 2.2) / speed + 12.0
        stats = run_robots(world, GroundTruthDetector(world).detect, duration, roles, clear_and_sort=clear_and_sort)
    finally:
        world.close()
    print(f"{name}: {stats.sorted_correctly} sorted, {len(stats.arm_contacts)} contacts, closest "
          f"{stats.min_arm_distance * 100:.1f} cm, {stats.arm_waits} waits, failures {dict(stats.failures)}")
    assert stats.arm_contacts == [] and stats.min_arm_distance > 0


def side_by_side(check: bool):
    """Robot 2 moves to a box at x = 0.05 (2 s) while Robot 1 should pick the box at x = -0.05 right next to it:
    with the check (a PickPlaceTask) or without it (Robot 1 follows its planned path regardless). Returns the
    closest approach of the arms [m] and the task (None without the check)."""
    world = World(ConveyorConfig(speed=0.0, max_boxes=0), physics=True)
    try:
        box1 = world.conveyor.spawn_box(BoxColor.RED, x=-0.05, y=0.0)
        box2 = world.conveyor.spawn_box(BoxColor.RED, x=0.05, y=0.0)
        world.run(0.5)
        arm2 = follow_path(world, plan_pick(world.C, 2, box2).path, 2.0, robot_id=2)
        next(arm2)
        if check:
            track = Track(0, BoxColor.RED, world.C.getFrame(box1).getPosition().copy(), np.zeros(3), world.time)
            task = PickPlaceTask(world, Assignment(1, 0, "region_A", world.time + 2.0, track.position), track,
                                 pick_only=True)
            arm1 = iter(task.advance, False)
        else:
            task, arm1 = None, follow_path(world, plan_pick(world.C, 1, box1).path, 2.0, robot_id=1)
        closest = float("inf")
        for _ in range(60):                                   # 3 s
            next(arm1, None)
            next(arm2, None)
            world.step()
            closest = min(closest, arm_proximity(world.C)[0])
        return closest, task
    finally:
        world.close()


def test_a_pick_into_the_moving_other_arm_is_given_up():
    unchecked, _ = side_by_side(check=False)
    checked, task = side_by_side(check=True)
    print(f"without the check the arms come {unchecked * 100:.1f} cm close, with it {checked * 100:.1f} cm; "
          f"Robot 1: {task.result.failed_stage}, {task.waits} wait(s)")
    assert unchecked < 0                                       # they would collide
    assert task.result.failed_stage == "pick" and task.waits == 1 and checked > 0
