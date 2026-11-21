import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.planning.pick import ARM_JOINT_NAMES, own_arm_only, plan_pick
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.scene import ROBOT_PREFIX, arm_joints
from conveyor_sort.sim.world import World


def world_with_box(x: float, y_offset: float = 0.0):
    """Stopped belt with one box at belt coordinate x."""
    world = World(ConveyorConfig(speed=0.0, spacing=100.0, lateral_jitter=0.0))
    name = world.conveyor.spawn_box(BoxColor.RED)
    frame = world.C.getFrame(name)
    frame.setRelativePosition([x, y_offset, frame.getRelativePosition()[2]])
    return world, name


@pytest.mark.parametrize("robot_id, x, y_offset", [
    (1, -0.5, 0.0), (1, -0.4, 0.1), (1, -0.1, -0.1), (2, 0.5, 0.05), (2, -0.4, 0.0),
])
def test_pick_plan_reaches_box_without_collisions(robot_id, x, y_offset):
    world, box = world_with_box(x, y_offset)
    C = world.C
    q_start = C.getJointState()
    plan = plan_pick(C, robot_id, box)
    assert plan.feasible
    assert plan.planning_time < 2.0
    assert np.abs(plan.path[0] - q_start).max() < 0.3            # starts next to the current state

    other = slice(7, 14) if robot_id == 1 else slice(0, 7)       # the other arm does not move
    assert np.abs(plan.path[:, other] - q_start[other]).max() < 1e-3

    for q in plan.path:                                          # every waypoint is collision-free
        C.setJointState(q)
        C.computeCollisions()
        assert C.getCollisions(0.0) == []

    gripper = C.getFrame(f"{ROBOT_PREFIX[robot_id]}gripper")     # final pose: at the box, pointing down
    assert gripper.getPosition() == pytest.approx(C.getFrame(box).getPosition(), abs=0.003)
    assert gripper.getRotationMatrix()[:, 2] == pytest.approx([0, 0, 1], abs=0.02)


def test_pick_plan_is_infeasible_out_of_reach():
    world, box = world_with_box(0.8)       # far right: Robot 1 cannot reach
    assert not plan_pick(world.C, 1, box).feasible


def test_own_arm_only_keeps_the_other_arm_where_it_is():
    """#141: only the planning robot's joints are executed, so a plan must not move the other arm out of the way.
    Robot 2 reaches for a box, Robot 1 plans a pick of the same box: within own_arm_only Robot 2's joints stay
    fixed and the plan is infeasible (Robot 2 is in the way). Planned with both arms, it was "feasible" with Robot
    2 moved by 0.14 rad, and executed with Robot 2 where it is the path went 14 cm into Robot 2."""
    world = World(ConveyorConfig(speed=0.0, max_boxes=0))
    C = world.C
    box = world.conveyor.spawn_box(BoxColor.GREEN, x=-0.1, y=0.0)
    reach = plan_pick(C, 2, box)
    assert reach.feasible
    C.setJointState(reach.path[12])                               # Robot 2 on its way down to the box
    q, other = C.getJointState(), arm_joints(C, 2)
    with own_arm_only(C):
        plan = plan_pick(C, 1, box)
    assert np.array_equal(plan.path[:, other], np.tile(q[other], (len(plan.path), 1)))
    assert np.array_equal(C.getJointState(), q) and C.getJointNames() == ARM_JOINT_NAMES   # C restored
    assert not plan.feasible
