import numpy as np
import pytest

from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.planning.pick_place import PickPlaceTask, pick_and_place
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.regions import region_contains
from conveyor_sort.sim.world import World
from conveyor_sort.sim.scene import HOME_Q
from test_moving_pick import tracked_box


def assignment_for(track, robot_id, region, pick_time):
    return Assignment(robot_id, track.track_id, region, pick_time, track.predict(pick_time))


@pytest.mark.parametrize("robot_id, speed, lead_time, region", [
    (1, 0.1, 3.0, "region_A"),
    (1, 0.2, 2.5, "region_B"),
    (2, 0.3, 3.5, "bin_gray"),
    (2, 0.3, 3.0, "region_B"),
])
def test_assignment_is_carried_out_for_either_robot(robot_id, speed, lead_time, region):
    world, box, track = tracked_box(speed)
    C = world.C
    collisions = []

    def check(world):
        C.computeCollisions()
        collisions.extend(C.getCollisions(0.0))

    result = pick_and_place(world, assignment_for(track, robot_id, region, world.time + lead_time), track, check)

    assert result.success and result.failed_stage is None
    assert result.box == box
    assert 0 < result.planning_time < 5.0
    assert region_contains(C, region, C.getFrame(box).getPosition())
    assert C.getFrame(box).getParent().name == region
    assert C.getJointState() == pytest.approx(list(HOME_Q) * 2, abs=1e-3)   # both arms at home
    assert world.conveyor.missed == 0
    assert collisions == []
    print(f"robot {robot_id}, {speed} m/s: planning {result.planning_time:.2f} s, total sim time {world.time:.1f} s")


def test_unreachable_assignment_fails_at_pick_without_moving():
    world, box, track = tracked_box(speed=0.1)
    q = world.C.getJointState()
    time = world.time
    result = pick_and_place(world, assignment_for(track, 2, "bin_gray", world.time + 2.0), track)
    assert not result.success and result.failed_stage == "pick" and result.box is None
    assert world.C.getJointState() == pytest.approx(q)
    assert world.time == time
    assert box in world.conveyor.on_belt()


def test_missed_grasp_sends_the_arm_home():
    world, box, track = tracked_box(speed=0.1)
    track.position = track.position + [0, 0.03, 0]          # tracker is 3 cm off: gripper arrives beside the box
    result = pick_and_place(world, assignment_for(track, 1, "region_A", world.time + 3.0), track)
    assert not result.success and result.failed_stage == "grasp"
    assert world.C.getJointState()[:7] == pytest.approx(HOME_Q, abs=1e-3)
    assert box in world.conveyor.on_belt()


def test_task_moves_only_its_own_arm():
    world, box, track = tracked_box(speed=0.3)
    task = PickPlaceTask(world, assignment_for(track, 2, "bin_gray", world.time + 3.5), track)
    while task.advance():
        assert world.C.getJointState()[:7] == pytest.approx(HOME_Q)      # Robot 1 is not touched
        world.step()
    assert task.result.success and task.result.box == box


def test_pick_is_not_started_when_the_lift_from_the_grasp_pose_cannot_be_planned():
    """Far side of the belt at the upstream limit: the grasp pose is reachable, but the arm cannot lift from it."""
    world = World(ConveyorConfig(speed=0.05, max_boxes=0))
    box = world.conveyor.spawn_box(BoxColor.BLUE, x=-0.87 - 3.0 * 0.05, y=0.07)
    track = Track(0, BoxColor.BLUE, world.C.getFrame(box).getPosition(), np.array([0.05, 0, 0]), 0.0)
    q = world.C.getJointState()
    result = pick_and_place(world, assignment_for(track, 1, "region_C", 3.0), track)
    assert not result.success and result.failed_stage == "pick" and result.box is None
    assert world.time == 0.0 and world.C.getJointState() == pytest.approx(q)      # nothing moved
    world.run(3.0)                                                                # 15 cm further on it works
    track = Track(0, BoxColor.BLUE, world.C.getFrame(box).getPosition(), np.array([0.05, 0, 0]), world.time)
    assert pick_and_place(world, assignment_for(track, 1, "region_C", world.time + 3.0), track).success
