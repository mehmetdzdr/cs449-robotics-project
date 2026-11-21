import numpy as np
import pytest

from conveyor_sort.coordination.allocation import COLORED, FIXED_ROLES
from conveyor_sort.coordination.clear_and_sort import (CLEAR_DELAY, CLEAR_XS, blockers, destination,
                                                       plan_clear_and_sort, plan_clearing_job)
from conveyor_sort.interfaces import BoxColor, Track
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.regions import BUFFER
from conveyor_sort.sim.scenarios import make_scenario
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
    assert plan_clear_and_sort(world, far, seen, {}, (BoxColor.RED,)) is not None              # far half: Robot 2's side (#154)
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

    stats = run_robots(world, GroundTruthDetector(world).detect, duration=60.0, on_step=check, **kwargs)
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
                 (0.08, 0.07, 0.0, BoxColor.RED), (0.03, -0.065, 0.0, BoxColor.BLUE), (0.08, 0.09, -0.06, BoxColor.GREEN),
                 (0.05, 0.08, 0.05, BoxColor.RED)]   # far half of the belt: Robot 2's side since #154
    outcomes = []
    for speed, gap, y, color in scenarios:
        world, _, _ = blocked_scene(speed, gap, y, color)
        stats, collisions = run(world, clear_and_sort=True)
        assert collisions == []
        outcomes.extend(stats.clear_and_sort)
    print(f"clear-and-sort success rate: {sum(outcomes)} of {len(outcomes)}")
    assert outcomes == [True] * len(scenarios)


@pytest.mark.parametrize("speed, gap, y", [(0.1, 0.08, 0.0)])     # belt too fast
def test_outside_the_limits_the_boxes_are_left_alone(speed, gap, y):
    world, _, _ = blocked_scene(speed, gap, y)
    stats, collisions = run(world, clear_and_sort=True)
    assert stats.clear_and_sort == [] and stats.events == [] and stats.missed == 2
    assert collisions == []


# Physics backend: clearing by whichever robot can reach the blocker (#101). Planning only (kinematic World: the
# reachability checks are KOMO IK); the physical runs are in test_loops_physics.py.

def cluster(x, speed, target_color=BoxColor.RED, time=5.0):
    """A target with a gray box touching it in front and one touching it on the far side: both grasp axes are
    blocked, one removal frees one of them."""
    d = 0.061
    return [track(0, target_color, x, speed=speed, time=time), track(1, BoxColor.GRAY, x + d, speed=speed, time=time),
            track(2, BoxColor.GRAY, x, y=0.2 + d, speed=speed, time=time)]


def decide(tracks, roles, buffer_free=6):
    world = World(ConveyorConfig(max_boxes=0))
    world.time = 5.0
    return plan_clearing_job(world, tracks, {t.track_id: 0.0 for t in tracks}, {}, roles, buffer_free)


def plan(tracks, roles, buffer_free=6):
    return decide(tracks, roles, buffer_free).job


def test_destination_is_the_own_region_only():
    """Robots on opposite sides (#154): no region and no buffer is reachable by both, a robot only takes a blocker
    of its own colors."""
    assert destination(1, BoxColor.GRAY) is None and destination(2, BoxColor.GRAY) == "bin_gray"
    assert destination(2, BoxColor.RED) is None and destination(2, BoxColor.GREEN) is None
    assert destination(1, BoxColor.BLUE) == "region_C"


def test_gray_blocker_goes_to_robot_2_and_the_target_to_robot_1():
    job = plan(cluster(-0.4, 0.03), FIXED_ROLES)
    assert job is not None
    for step in job.steps:
        assert (step.robot_id, step.region) == ((2, "bin_gray") if step.track_id in (1, 2) else (1, "region_A"))


def test_robot_1_alone_cannot_clear_gray_blockers():
    assert plan(cluster(-0.4, 0.03), {1: COLORED}) is None     # it cannot take a gray box anywhere


@pytest.mark.parametrize("x, target_color", [(0.0, BoxColor.GREEN), (0.3, BoxColor.GREEN)])
def test_gray_blocker_robot_2_can_reach_goes_straight_into_the_bin(x, target_color):
    job = plan(cluster(x, 0.01, target_color), FIXED_ROLES)
    assert job is not None
    assert all(step.region != BUFFER for step in job.steps)
    assert all((s.robot_id, s.region) == (2, "bin_gray") for s in job.steps if s.track_id in (1, 2))


def test_target_robot_1_cannot_reach_in_robot_2_only_zone_is_reported():
    decision = decide(cluster(0.6, 0.01), {1: COLORED})
    assert decision.job is None and decision.infeasible == [0]


def test_clearing_that_cannot_finish_in_time_is_not_started():
    surrounded = cluster(-0.7, 0.1) + [track(3, BoxColor.GRAY, -0.761, speed=0.1, time=5.0),
                                       track(4, BoxColor.GRAY, -0.7, y=0.139, speed=0.1, time=5.0)]
    assert decide(surrounded, {1: COLORED}).infeasible == [0]      # two removals: the red box would be out of reach
    assert decide(cluster(-0.7, 0.1), {1: COLORED}).infeasible == [0]   # one removal, 0.1 m/s: out of reach after 12.5 s


# Time limits (#102): the whole schedule is checked before the first removal, with the earliest start that works.

def layout(blockers, x, speed):
    """A red target at `x` with 1 (in front), 2 (in front, far side) or 3 (in front, behind, far side) gray boxes
    touching it. 1: grasped across, no removal; 2 and 3: one removal frees an axis."""
    d = 0.061
    offsets = {1: [(d, 0)], 2: [(d, 0), (0, d)], 3: [(d, 0), (-d, 0), (0, d)]}[blockers]
    return [track(0, BoxColor.RED, x, speed=speed, time=5.0)] + [
        track(i + 1, BoxColor.GRAY, x + dx, y=0.2 + dy, speed=speed, time=5.0) for i, (dx, dy) in enumerate(offsets)]


@pytest.mark.parametrize("speed, blockers, removals", [
    (0.05, 1, 0), (0.05, 2, 1), (0.05, 3, 1),
    (0.1, 1, 0), (0.1, 2, None), (0.1, 3, None),        # None: reported infeasible (target out of reach after 15.5 s)
    (0.2, 1, 0), (0.2, 2, None), (0.2, 3, None),
])
def test_clearing_is_started_only_if_it_finishes_in_time(speed, blockers, removals):
    decision = decide(layout(blockers, -0.45, speed), {1: COLORED, 2: ()})   # Robot 2 removes the gray boxes (#154)
    if removals is None:
        assert decision.job is None and decision.infeasible == [0] and not decision.later
    else:
        assert decision.job.target == 0 and len(decision.job.steps) == removals + 1
        assert decision.job.steps[-1].across_belt or blockers != 1     # one box in front only: grasped across


def test_clearing_that_can_start_later_is_deferred():
    """Robot 2 alone, a gray target behind a gray box (and one on the far side), upstream of Robot 2's reach: the
    blocker can only be taken to the bin once it is within reach, so the clearing starts later."""
    tracks = [track(0, BoxColor.GRAY, -0.9, time=5.0), track(1, BoxColor.GRAY, -0.839, time=5.0),
              track(2, BoxColor.GRAY, -0.9, y=0.261, time=5.0)]
    decision = decide(tracks, {2: (BoxColor.GRAY,)})
    assert decision.job is None and decision.infeasible == [] and 5.0 < decision.later[0] < 25.0


def test_cluster_that_cannot_be_cleared_in_any_order_is_reported():
    scenario = make_scenario("cluster_3x3", spacing="touching", x=-0.8)
    tracks = [track(0, BoxColor.RED, scenario.target[0], y=scenario.target[1], time=5.0)] + [
        track(i + 1, color, p[0], y=p[1], time=5.0) for i, (color, p) in enumerate(scenario.neighbours)]
    assert decide(tracks, {1: COLORED}).infeasible == [0]
