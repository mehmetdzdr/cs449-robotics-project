"""Disturbed boxes (#104): the tracker follows a displaced box and the pick is re-planned from the new prediction."""
import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.planning.pick_place import PickPlaceTask
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.scene import HOME_Q, arm_joints
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots

pytestmark = pytest.mark.slow   # physics scenarios in real time: nightly / on demand in CI (#97)


def approaching(world) -> bool:
    """Robot 1's arm has left its home pose (it is on its way to a box)."""
    return np.abs(world.C.getJointState()[arm_joints(world.C, 1)] - HOME_Q).max() > 0.05


def sort_with_disturbance(disturb, boxes=((BoxColor.RED, -0.9),), duration=25.0, observe=True):
    """Robot 1 sorts red boxes from a belt at 0.05 m/s; `disturb(world, names, since)` is called every step with the
    time since the arm started moving (None before). Returns the stats and the box names."""
    world = World(ConveyorConfig(speed=0.05, max_boxes=0), physics=True)
    try:
        names = [world.conveyor.spawn_box(color, x=x, y=0.0) for color, x in boxes]
        start, offsets = [None], {}

        def on_step(world):
            if start[0] is None and approaching(world):
                start[0] = world.time
            disturb(world, names, None if start[0] is None else world.time - start[0])
            for box in world.conveyor.held - set(offsets):   # where the gripper met the box when it closed on it
                offsets[box] = np.linalg.norm(world.C.getFrame(box).getPosition()[:2]
                                              - world.C.getFrame("l_gripper").getPosition()[:2])

        stats = run_robots(world, GroundTruthDetector(world).detect, duration, {1: (BoxColor.RED,)}, on_step=on_step)
        stats.grasp_offsets = offsets
        held = bool(world.conveyor.held)
        return stats, names, held
    finally:
        world.close()


def shift(world, box, offset, yaw_deg=0.0):
    position = world.C.getFrame(box).getPosition() + offset
    half = np.radians(yaw_deg) / 2
    world.conveyor._set_pose(box, position, [np.cos(half), 0, 0, np.sin(half)])


def bump_forward(world, names, since, done=[]):
    if since is not None and since > 0.5 and not done:   # early in the 3 s approach
        shift(world, names[0], [0.03, 0, 0])               # pushed 3 cm along the belt: the fingers close along it
        done.append(True)


@pytest.mark.parametrize("observe", [True, False])
def test_box_bumped_during_the_approach_is_aimed_at_again(observe, monkeypatch):
    """Pushed 3 cm along the belt early in the approach. With re-planning the gripper meets the box where it now is
    (measured 2.0 and 3.3 mm off); without, the first grasp fails or the gripper is well off (measured 13.8 and
    28.3 mm, depending on how far the approach had got). A box missed by up to 3 cm can still be sorted here: the belt
    moves boxes by setting their pose until the grasp, so a finger that lands on the box is pushed out of it and the
    squeeze centres the box; a real belt would not forgive it."""
    if not observe:
        monkeypatch.setattr(PickPlaceTask, "observe", lambda self, tracks: None)
    stats, names, held = sort_with_disturbance(lambda w, n, s, done=[]: bump_forward(w, n, s, done))
    first = stats.events[0]
    offset = stats.grasp_offsets.get(names[0], float("inf"))   # first time the gripper closed on the box
    print(f"re-plan {'on' if observe else 'off'}: {[(e.failed_stage, round(e.time, 1)) for e in stats.events]}, "
          f"gripper {offset * 1000:.1f} mm from the box at the grasp")
    if observe:
        assert first.correct and offset < 0.012                 # aimed at the box where it is now
    else:
        assert first.failed_stage == "grasp" or offset > 0.01   # aimed at where it was
    assert stats.sorted_correctly == 1 and not held


def test_box_rotated_during_the_approach_is_still_grasped():
    def rotate(world, names, since, done=[]):
        if since is not None and since > 0.5 and not done:
            shift(world, names[0], [0, 0, 0], yaw_deg=30)  # turned by a collision, same place
            done.append(True)

    stats, _, held = sort_with_disturbance(rotate)
    print([(e.failed_stage, round(e.time, 1)) for e in stats.events])
    assert stats.events[0].correct and not held


def test_box_pushed_off_the_belt_is_given_up():
    def push_off(world, names, since, done=[]):
        if since is not None and since > 0.1 and not done:
            p = world.C.getFrame(names[0]).getPosition()
            world.conveyor._set_pose(names[0], [p[0], 0.5, 0.63], [1.0, 0, 0, 0])   # onto the table beyond the belt
            done.append(True)

    stats, _, held = sort_with_disturbance(push_off)
    print([(e.failed_stage, round(e.time, 1)) for e in stats.events])
    assert [e.failed_stage for e in stats.events] == ["lost"] and not held


def test_neighbour_bumped_during_a_grasp_is_followed_and_sorted():
    """While Robot 1 squeezes the first box, the one 30 cm behind it is pushed 2 cm sideways: the tracker follows
    it (same track) and it is sorted next."""
    def bump_neighbour(world, names, since, done=[]):
        if names[0] in world.conveyor.held and not done:
            shift(world, names[1], [0, 0.02, 0])
            done.append(True)

    stats, _, held = sort_with_disturbance(bump_neighbour, boxes=((BoxColor.RED, -0.6), (BoxColor.RED, -0.9)),
                                           duration=40.0)
    print([(e.failed_stage, round(e.time, 1)) for e in stats.events])
    assert stats.sorted_correctly == 2 and stats.failed == 0 and not held
