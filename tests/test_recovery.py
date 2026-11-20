"""Recovery from missed grasps, slips and drops (#105), physics backend."""
import numpy as np
import pytest

from conveyor_sort.interfaces import Assignment, BoxColor, Detection, Track
from conveyor_sort.physics.friction_grasp import FINGER_OPEN
from conveyor_sort.planning.pick_place import PickPlaceTask
from conveyor_sort.sim.conveyor import BOX_SIZE, ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.regions import REGIONS, region_contains
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import MAX_ATTEMPTS, run_robots

pytestmark = pytest.mark.slow   # physics scenarios in real time: nightly / on demand in CI (#97)


def no_box_in_a_gripper(world) -> bool:
    fingers = world.C.getJointState()[list(world.fingers.values())]
    return not world.conveyor.held and bool(np.all(fingers > 0.035))


def test_missed_grasp_is_retried():
    """The detector reports the box 4.5 cm off across the belt for the first 6 s: the first grasp closes on
    nothing; with correct detections afterwards the box is picked again and sorted."""
    world = World(ConveyorConfig(speed=0.05, max_boxes=0), physics=True)
    try:
        world.conveyor.spawn_box(BoxColor.RED, x=-0.9, y=0.0)
        truth = GroundTruthDetector(world).detect

        def detect() -> list[Detection]:
            offset = [0, 0.045, 0] if world.time < 6.0 else [0, 0, 0]
            return [Detection(d.color, d.position + offset, d.confidence, d.timestamp) for d in truth()]

        stats = run_robots(world, detect, 30.0, {1: (BoxColor.RED,)})
        print(f"events {[(e.failed_stage, round(e.time, 1)) for e in stats.events]}, failures {dict(stats.failures)}")
        assert [e.failed_stage for e in stats.events] == ["grasp", None] and stats.sorted_correctly == 1
        assert stats.failures == {"grasp": 1} and no_box_in_a_gripper(world)
    finally:
        world.close()


def test_slip_with_low_friction_is_retried_then_counted():
    """With friction 0.1 a grasp slips during the lift (0.05: the squeeze shoots the box off the table, 0.2: it is
    held). The box falls back onto the belt and is tried again, up to MAX_ATTEMPTS attempts while it is reachable;
    every failure is counted with where the box came to rest. Friction 0.1 is near the limit, so the attempts end
    differently from run to run (seen: slip then sorted; slip, slip onto the table; slip, slip, held then dropped
    in transport; slip, slip, then out of reach; slip, then held but set down badly: "release", #146). What holds in
    every run: no box is tried endlessly, every failure is counted, every slipped or dropped box is located."""
    world = World(ConveyorConfig(speed=0.05, max_boxes=0), physics=True, friction=0.1)
    try:
        world.conveyor.spawn_box(BoxColor.RED, x=-0.9, y=0.0)
        stats = run_robots(world, GroundTruthDetector(world).detect, 40.0, {1: (BoxColor.RED,)})
        print(f"events {[(e.failed_stage, e.dropped_on, round(e.time, 1)) for e in stats.events]}")
        assert 1 <= len(stats.events) <= MAX_ATTEMPTS                       # not tried endlessly (one run: 4)
        failed = [e for e in stats.events if not e.success]
        assert set(stats.failures) <= {"grasp", "slip", "lift", "drop", "place", "release"}   # what friction can do
        assert sum(stats.failures.values()) == len(failed)                  # every failure counted
        assert all(e.dropped_on for e in failed if e.failed_stage in ("slip", "drop"))   # and located
        assert no_box_in_a_gripper(world)
    finally:
        world.close()


def test_every_attempt_slips_then_the_box_is_given_up():
    """The test opens the fingers right after every lift: the box falls back onto the belt each time and is tried
    again until MAX_ATTEMPTS, then left alone. The box is not seen while held, so its track is dropped and a new
    one starts when it is back: the attempt count must carry over."""
    world = World(ConveyorConfig(speed=0.05, max_boxes=0), physics=True)
    try:
        box = world.conveyor.spawn_box(BoxColor.RED, x=-0.9, y=0.0)

        def let_it_slip(world):
            if box in world.conveyor.held and world.C.getFrame(box).getPosition()[2] > 0.62 + BOX_SIZE / 2 + 0.05:
                world.arms.set_fingers(1, FINGER_OPEN, 0.1)

        stats = run_robots(world, GroundTruthDetector(world).detect, 45.0, {1: (BoxColor.RED,)}, on_step=let_it_slip)
        print(f"events {[(e.failed_stage, e.dropped_on, round(e.time, 1)) for e in stats.events]}")
        assert stats.failures == {"slip": MAX_ATTEMPTS} and stats.drops == {"belt": MAX_ATTEMPTS}
        assert len(stats.events) == MAX_ATTEMPTS and no_box_in_a_gripper(world)
    finally:
        world.close()


def drop(where: str):
    """Robot 1 picks a box from the stopped belt for region A; the test opens the fingers on the way: right after
    the place motion starts ("belt"), once the box is past the belt edge ("table"), or after the lift with another
    box put right under it ("box"). Returns (result, final box position, the arm is free)."""
    world = World(ConveyorConfig(speed=0.0, max_boxes=0), physics=True)
    try:
        box = world.conveyor.spawn_box(BoxColor.RED, x=-0.3, y=0.0)
        other = world.conveyor.spawn_box(BoxColor.GRAY, x=-0.9, y=0.0) if where == "box" else None
        world.run(0.5)
        start = world.C.getFrame(box).getPosition().copy()
        track = Track(0, BoxColor.RED, start, np.zeros(3), world.time)
        pick_time = world.time + 2.5
        task = PickPlaceTask(world, Assignment(1, 0, "region_A", pick_time, track.predict(pick_time)), track)
        released = lifted = False
        while task.advance():
            world.step()
            p = world.C.getFrame(box).getPosition()
            lifted = lifted or p[2] > start[2] + 0.1          # once lifted (the place descends again later)
            if released or not lifted:
                continue
            if where == "belt" and np.linalg.norm(p[:2] - start[:2]) > 0.03:
                released = True
            elif where == "table" and p[1] < world.conveyor.geometry.y - world.conveyor.geometry.width / 2 - 0.05 \
                    and not any(region_contains(world.C, r, p) for r in REGIONS):
                released = True
            elif where == "box":
                world.conveyor._set_pose(other, [p[0], p[1], start[2]], [1.0, 0, 0, 0])   # right under the lifted box
                world.run(0.2)
                released = True
            if released:
                world.arms.set_fingers(1, FINGER_OPEN, 0.1)
        world.run(1.0)
        return task.result, world.C.getFrame(box).getPosition().copy(), no_box_in_a_gripper(world)
    finally:
        world.close()


@pytest.mark.parametrize("where", ["belt", "table", "box"])
def test_drop_is_detected_and_located(where):
    result, position, arm_free = drop(where)
    print(f"dropped above the {where}: {result.failed_stage}, landed on {result.dropped_on}, z {position[2]:.3f}")
    assert result.failed_stage in ("drop", "slip") and not result.success
    assert result.dropped_on == where and arm_free
    if where == "box":
        assert position[2] > 0.62 + BOX_SIZE + 0.02     # resting on the other box (0.71 m)
