import numpy as np
import pytest

from conveyor_sort.planning.execute import execute_path, grasp
from conveyor_sort.planning.pick import plan_lift, plan_pick
from conveyor_sort.sim.scene import ROBOT_PREFIX
from test_pick import world_with_box


@pytest.mark.parametrize("robot_id, x, y_offset", [(1, -0.7, 0.0), (1, -0.3, 0.1), (2, 0.2, -0.1), (2, 0.6, 0.05)])
def test_pick_grasp_and_lift(robot_id, x, y_offset):
    world, box = world_with_box(x, y_offset)
    C = world.C
    start = C.getFrame(box).getPosition()
    collisions = []

    def check(world):
        C.computeCollisions()
        collisions.extend(C.getCollisions(0.0))

    pick = plan_pick(C, robot_id, box)
    assert pick.feasible
    execute_path(world, pick.path, 2.0, check)
    assert world.time == pytest.approx(2.0)
    assert C.getJointState() == pytest.approx(pick.path[-1])

    assert grasp(world, robot_id, box)
    assert box not in world.conveyor.on_belt()
    assert C.getFrame(box).getParent().name == f"{ROBOT_PREFIX[robot_id]}gripper"

    lift = plan_lift(C, robot_id, height=0.15)
    assert lift.feasible
    execute_path(world, lift.path, 1.0, check)
    assert C.getFrame(box).getPosition() == pytest.approx(start + [0, 0, 0.15], abs=0.005)   # box went up with the gripper
    assert collisions == []


def test_grasp_fails_when_gripper_is_not_at_the_box():
    world, box = world_with_box(-0.5)
    assert not grasp(world, 1, box)
    assert box in world.conveyor.on_belt()

