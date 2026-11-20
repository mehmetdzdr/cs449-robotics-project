import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.planning.pick import plan_pick
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.scene import ROBOT_PREFIX
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
