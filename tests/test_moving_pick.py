import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.planning.pick import plan_lift, plan_moving_pick
from conveyor_sort.prediction.tracker import Tracker
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.world import World


def tracked_box(speed: float, observe: float = 1.0, x: float = -0.75):
    """World with one box moving from belt position x (the robots, at x = 0, reach from about -0.6), observed for
    `observe` seconds with the ground-truth detector."""
    world = World(ConveyorConfig(speed=speed, max_boxes=0))
    world.conveyor.spawn_box(BoxColor.RED, x=x, y=0.0)
    detector, tracker = GroundTruthDetector(world), Tracker()
    tracks = []
    world.run(observe, lambda w: tracks.__setitem__(slice(None), tracker.update(detector.detect(), w.time)))
    (track,) = tracks
    (box,) = world.conveyor.on_belt()
    return world, box, track


@pytest.mark.parametrize("speed, lead_time", [(0.05, 5.0), (0.1, 3.0), (0.2, 2.5), (0.3, 2.0)])
def test_moving_box_is_picked_at_the_predicted_intercept(speed, lead_time):
    world, box, track = tracked_box(speed)
    C = world.C
    assert track.velocity == pytest.approx([speed, 0, 0], abs=1e-6)
    collisions = []
    pick_time = world.time + lead_time
    plan = plan_moving_pick(C, 1, track, world.time, pick_time)
    assert plan.feasible
    start = world.time
    for k, waypoint in enumerate(plan.path, 1):  # the plan at its own times (the box moves on with the belt meanwhile)
        while world.time < start + k * lead_time / len(plan.path) - 1e-9:
            world.step()
        C.setJointState(waypoint)
        C.computeCollisions()
        collisions.extend(C.getCollisions(0.0))
    assert world.time == pytest.approx(pick_time)

    box_position = C.getFrame(box).getPosition()
    assert box_position == pytest.approx(track.predict(pick_time), abs=1e-6)                 # box is where predicted
    assert C.getFrame("l_gripper").getPosition() == pytest.approx(box_position, abs=0.003)   # and so is the gripper
    assert plan_lift(C, 1).feasible                                   # the arm can lift from the grasp pose
    assert collisions == []


def test_infeasible_when_box_is_outside_the_robots_reach_at_pick_time():
    world, _, track = tracked_box(speed=0.1, x=-1.1)
    # after 2 s the box is still at the far left of the belt (x = -0.8): Robot 2 cannot reach it
    assert not plan_moving_pick(world.C, 2, track, world.time, world.time + 2.0).feasible
    # after 18 s (x = 0.8) it has passed Robot 1's reach
    assert not plan_moving_pick(world.C, 1, track, world.time, world.time + 18.0).feasible


def test_planning_leaves_no_helper_frame_behind():
    world, _, track = tracked_box(speed=0.1)
    names = world.C.getFrameNames()
    plan_moving_pick(world.C, 1, track, world.time, world.time + 3.0)
    assert world.C.getFrameNames() == names


def test_paths_are_executed_on_physics_only():
    """#114: no kinematic path playback (joint states written per step) any more."""
    from conveyor_sort.planning.execute import follow_path
    world, _, track = tracked_box(speed=0.1)
    plan = plan_moving_pick(world.C, 1, track, world.time, world.time + 3.0)
    with pytest.raises(ValueError, match="physics"):
        next(follow_path(world, plan.path, 3.0, 1))
