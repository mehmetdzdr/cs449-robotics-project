"""KOMO paths executed by BotOp's controller for both arms (#90)."""
import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.physics.arms import ARM_JOINTS
from conveyor_sort.planning.execute import execute_path, follow_path
from conveyor_sort.planning.pick import plan_pick
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.scene import ROBOT_PREFIX
from conveyor_sort.sim.world import World


@pytest.fixture
def physics_world():
    worlds = []

    def make(boxes) -> tuple[World, list[str]]:
        """Stopped belt, boxes at (x, y offset); hyperSpeed 1 (fine steps for the tracking measurements)."""
        world = World(ConveyorConfig(speed=0.0, max_boxes=0), physics=True)
        worlds.append(world)
        names = [world.conveyor.spawn_box(BoxColor.RED, x=x, y=y) for x, y in boxes]
        world.run(0.5)                                     # boxes settle
        return world, names

    yield make
    for world in worlds:
        world.close()


def planned_gripper(world, robot_id, q):
    """Gripper position for joint state q (C is restored; the next sync would overwrite it anyway)."""
    measured = world.C.getJointState()
    world.C.setJointState(q)
    position = world.C.getFrame(f"{ROBOT_PREFIX[robot_id]}gripper").getPosition().copy()
    world.C.setJointState(measured)
    return position


def gripper_error(world, robot_id, q_planned) -> float:
    actual = world.C.getFrame(f"{ROBOT_PREFIX[robot_id]}gripper").getPosition()
    return float(np.linalg.norm(actual - planned_gripper(world, robot_id, q_planned)))


@pytest.mark.parametrize("robot_id, x, y", [(1, -0.7, 0.0), (1, -0.3, 0.1), (2, 0.2, -0.1), (2, 0.6, 0.05)])
def test_tracking_error_at_the_end_of_a_pick_path(physics_world, robot_id, x, y):
    world, (box,) = physics_world([(x, y)])
    pick = plan_pick(world.C, robot_id, box)
    assert pick.feasible
    start = world.time
    execute_path(world, pick.path, 2.0)
    assert world.time == pytest.approx(start + 2.0, abs=world.dt)
    at_end = gripper_error(world, robot_id, pick.path[-1])
    world.run(0.5)
    settled = gripper_error(world, robot_id, pick.path[-1])
    print(f"robot {robot_id} box ({x}, {y}): gripper error {at_end * 1000:.1f} mm at the end of the path, "
          f"{settled * 1000:.1f} mm 0.5 s later")
    assert at_end < 0.01 and settled < 0.003


def test_both_arms_at_once_with_different_durations(physics_world):
    world, (box1, box2) = physics_world([(-0.5, 0.0), (0.6, 0.0)])
    pick1, pick2 = plan_pick(world.C, 1, box1), plan_pick(world.C, 2, box2)
    assert pick1.feasible and pick2.feasible
    start = world.time
    arm1 = follow_path(world, pick1.path, 2.0, robot_id=1)
    arm2 = None
    errors = {}
    while len(errors) < 2:
        if arm1 is not None and next(arm1, "done") == "done":
            errors[1], arm1 = gripper_error(world, 1, pick1.path[-1]), None
        if arm2 is None and 2 not in errors and world.time >= start + 0.5:
            arm2 = follow_path(world, pick2.path, 3.0, robot_id=2)          # starts 0.5 s later, takes 3 s
        if arm2 is not None and next(arm2, "done") == "done":
            errors[2], arm2 = gripper_error(world, 2, pick2.path[-1]), None
            end2 = world.time
        world.step()
    print(f"both arms: gripper error {errors[1] * 1000:.1f} / {errors[2] * 1000:.1f} mm, robot 2 done at "
          f"{end2 - start:.2f} s")
    assert end2 - start == pytest.approx(3.5, abs=2 * world.dt)
    assert errors[1] < 0.01 and errors[2] < 0.01


def robot1_tracking_error(world, box1, box2, disturb: bool) -> float:
    """Robot 1 follows a 3 s path; with `disturb`, Robot 2 gets a new path while Robot 1 is moving.
    Returns Robot 1's largest joint deviation from its own reference [rad]."""
    pick1 = plan_pick(world.C, 1, box1)
    pick2 = plan_pick(world.C, 2, box2)
    start = world.time
    arm1, arm2, errors = follow_path(world, pick1.path, 3.0, robot_id=1), None, []
    for _ in arm1:
        if disturb and arm2 is None and world.time >= start + 1.0:
            arm2 = follow_path(world, pick2.path, 1.5, robot_id=2)
        if arm2 is not None:
            next(arm2, None)
        world.step()
        reference = world.arms.reference(1, world.time + world._t0)
        errors.append(np.abs(world.C.getJointState()[ARM_JOINTS[1]] - reference).max())
    return float(max(errors))


def test_new_path_for_one_arm_does_not_disturb_the_other(physics_world):
    world, (box1, box2) = physics_world([(-0.5, 0.0), (0.6, 0.0)])
    alone = robot1_tracking_error(world, box1, box2, disturb=False)
    world2, (box1, box2) = physics_world([(-0.5, 0.0), (0.6, 0.0)])
    disturbed = robot1_tracking_error(world2, box1, box2, disturb=True)
    print(f"robot 1 tracking error {alone * 1000:.1f} mrad alone, {disturbed * 1000:.1f} mrad while robot 2 is re-planned")
    # measured: 82 to 85 mrad alone, 82 to 98 mrad disturbed. Most of it is BotOp's control latency (the arm
    # follows its spline 0.1 s late), which #91 compensates; this only guards against a large disturbance
    assert alone < 0.12 and disturbed < 0.12
