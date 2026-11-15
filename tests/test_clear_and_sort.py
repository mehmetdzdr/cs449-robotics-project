import numpy as np
import pytest

from conveyor_sort.coordination.clear_and_sort import CLEAR_DELAY, CLEAR_XS, blockers, plan_clear_and_sort
from conveyor_sort.interfaces import BoxColor, Track
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots


def track(track_id, color, x, y=0.2, speed=0.05, time=0.0):
    return Track(track_id, color, np.array([x, y, 0.65]), np.array([speed, 0, 0]), time)


def test_blockers_are_close_neighbours_in_the_same_lane():
    red = track(0, BoxColor.RED, -0.5)
    in_front, behind = track(1, BoxColor.GRAY, -0.42), track(2, BoxColor.GRAY, -0.60)
    beside, far = track(3, BoxColor.GRAY, -0.5, y=0.27), track(4, BoxColor.GRAY, -0.38)
    assert [t.track_id for t in blockers(red, [red, in_front, behind, beside, far], 0.0)] == [1, 2]
    slower = track(5, BoxColor.GRAY, -0.35, speed=0.02)                  # the red box catches up with it
    assert blockers(red, [red, slower], 0.0) == [] and blockers(red, [red, slower], 3.0) == [slower]


def test_plan_gives_robot_2_the_gray_box_first():
    world = World(ConveyorConfig(speed=0.05, max_boxes=0))
    world.time = 5.0
    red, gray = track(0, BoxColor.RED, -0.68, time=5.0), track(1, BoxColor.GRAY, -0.60, time=5.0)
    seen = {0: 0.0, 1: 0.0, 2: 0.0}
    clear, sort = plan_clear_and_sort(world, [red, gray], seen, {}, (BoxColor.RED,))
    assert (clear.robot_id, clear.track_id, clear.target_region) == (2, 1, "bin_gray")
    assert (sort.robot_id, sort.track_id, sort.target_region) == (1, 0, "region_A")
    assert any(clear.pick_position[0] == pytest.approx(x) for x in CLEAR_XS)
    assert clear.pick_time == pytest.approx(5.0 + (clear.pick_position[0] + 0.60) / 0.05)
    assert sort.pick_time == pytest.approx(clear.pick_time + CLEAR_DELAY)

    assert plan_clear_and_sort(world, [red, gray], seen, {}, (BoxColor.BLUE,)) is None        # red is not wanted
    assert plan_clear_and_sort(world, [red], seen, {}, (BoxColor.RED,)) is None               # nothing in the way
    blue = track(2, BoxColor.BLUE, -0.60, time=5.0)
    assert plan_clear_and_sort(world, [red, blue], seen, {}, (BoxColor.RED,)) is None         # the blocker is not gray
    far = [track(0, BoxColor.RED, -0.68, y=0.25, time=5.0), track(1, BoxColor.GRAY, -0.60, y=0.25, time=5.0)]
    assert plan_clear_and_sort(world, far, seen, {}, (BoxColor.RED,)) is None                  # far half of the belt
    late = [track(0, BoxColor.RED, -0.08, time=5.0), track(1, BoxColor.GRAY, 0.0, time=5.0)]
    assert plan_clear_and_sort(world, late, seen, {}, (BoxColor.RED,)) is None                # too late to clear


def blocked_scene(speed=0.05, gap=0.08, y=0.0, color=BoxColor.RED):
    """A wanted box with a gray box `gap` in front of it (behind it if negative), nothing else on the belt."""
    world = World(ConveyorConfig(speed=speed, max_boxes=0))
    wanted = world.conveyor.spawn_box(color, x=-0.9, y=y)
    gray = world.conveyor.spawn_box(BoxColor.GRAY, x=-0.9 + gap, y=y)
    return world, wanted, gray


def run(world, **kwargs):
    collisions = []

    def check(world):
        world.C.computeCollisions()
        collisions.extend(world.C.getCollisions(0.0))

    stats = run_robots(world, GroundTruthDetector(world).detect, duration=45.0, on_step=check, **kwargs)
    return stats, collisions


def test_scripted_blocking_scenario_succeeds():
    world, red, gray = blocked_scene()
    stats, collisions = run(world, clear_and_sort=True)
    assert stats.clear_and_sort == [True]
    assert [(e.robot_id, e.color, e.region, e.correct) for e in stats.events] == [
        (2, BoxColor.GRAY, "bin_gray", True), (1, BoxColor.RED, "region_A", True)]       # gray first, then red
    assert world.C.getFrame(gray).getParent().name == "bin_gray"
    assert world.C.getFrame(red).getParent().name == "region_A"
    assert stats.missed == 0 and collisions == []


def test_without_clear_and_sort_the_blocked_boxes_are_left_alone():
    world, red, gray = blocked_scene()
    stats, collisions = run(world)
    assert stats.events == [] and stats.clear_and_sort == [] and stats.missed == 2
    assert collisions == []


def test_success_rate_over_varied_blocking_scenarios():
    scenarios = [(0.05, 0.08, 0.0, BoxColor.RED), (0.05, -0.065, -0.06, BoxColor.BLUE), (0.03, 0.10, -0.06, BoxColor.GREEN),
                 (0.08, 0.07, 0.0, BoxColor.RED), (0.03, -0.065, 0.0, BoxColor.BLUE), (0.08, 0.09, -0.06, BoxColor.GREEN)]
    outcomes = []
    for speed, gap, y, color in scenarios:
        world, _, _ = blocked_scene(speed, gap, y, color)
        stats, collisions = run(world, clear_and_sort=True)
        assert collisions == []
        outcomes.extend(stats.clear_and_sort)
    print(f"clear-and-sort success rate: {sum(outcomes)} of {len(outcomes)}")
    assert outcomes == [True] * len(scenarios)


@pytest.mark.parametrize("speed, gap, y", [(0.1, 0.08, 0.0),      # belt too fast
                                           (0.05, 0.08, 0.05),    # gray box on the far half of the belt
                                           (0.08, -0.08, 0.0)])   # gray box behind: the wanted box is out of reach afterwards
def test_outside_the_limits_the_boxes_are_left_alone(speed, gap, y):
    world, _, _ = blocked_scene(speed, gap, y)
    stats, collisions = run(world, clear_and_sort=True)
    assert stats.clear_and_sort == [] and stats.events == [] and stats.missed == 2
    assert collisions == []
