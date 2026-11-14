import pytest

from conveyor_sort.planning.execute import execute_path, grasp, release
from conveyor_sort.planning.pick import plan_home, plan_lift, plan_pick, plan_place
from conveyor_sort.sim.conveyor import BOX_SIZE
from conveyor_sort.sim.regions import region_contains
from conveyor_sort.sim.scene import HOME_Q
from test_pick import world_with_box


@pytest.mark.parametrize("robot_id, x, region", [
    (1, -0.6, "region_A"), (1, -0.2, "region_B"), (1, -0.4, "region_C"), (2, 0.5, "bin_gray"), (2, 0.1, "region_B"),
])
def test_full_pick_and_place_of_a_stationary_box(robot_id, x, region):
    world, box = world_with_box(x)
    C = world.C
    collisions = []

    def check(world):
        C.computeCollisions()
        collisions.extend(C.getCollisions(0.0))

    def run(plan, duration):
        assert plan.feasible
        execute_path(world, plan.path, duration, check)

    run(plan_pick(C, robot_id, box), 2.0)
    assert grasp(world, robot_id, box)
    run(plan_lift(C, robot_id), 1.0)
    run(plan_place(C, robot_id, box, region), 3.0)
    release(world, box, region)
    run(plan_lift(C, robot_id, height=0.1), 1.0)     # retreat straight up so the fingers clear the box
    run(plan_home(C, robot_id), 2.0)

    position = C.getFrame(box).getPosition()
    assert region_contains(C, region, position)
    assert position[:2] == pytest.approx(C.getFrame(region).getPosition()[:2], abs=0.005)   # at the region centre
    assert position[2] == pytest.approx(0.6 + BOX_SIZE / 2, abs=0.005)                       # resting on the table
    assert C.getFrame(box).getParent().name == region
    assert C.getJointState()[(robot_id - 1) * 7: robot_id * 7] == pytest.approx(HOME_Q, abs=1e-3)
    assert collisions == []
