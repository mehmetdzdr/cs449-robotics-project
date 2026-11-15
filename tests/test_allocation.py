import numpy as np
import pytest

from conveyor_sort.coordination.allocation import FIXED_ROLES, LEAD_TIME, SHARED_ROLES, allocate
from conveyor_sort.interfaces import BoxColor, Track
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.regions import DEFAULT_TARGET
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots

SPEED = 0.1


def empty_world() -> World:
    world = World(ConveyorConfig(speed=SPEED, spacing=100.0))
    world.time = 10.0
    return world


def track(track_id: int, color: BoxColor, pick_x: float, world: World, y: float = 0.2) -> Track:
    """A box on the belt that will be at `pick_x` at the pick time."""
    return Track(track_id, color, np.array([pick_x - SPEED * LEAD_TIME, y, 0.65]), np.array([SPEED, 0, 0]), world.time)


def allocate_all(world, tracks, roles=SHARED_ROLES, free=(1, 2), blocked=None):
    return allocate(world, tracks, {t.track_id: 0.0 for t in tracks}, blocked or {}, roles, list(free))


def test_each_robot_gets_its_most_urgent_box():
    world = empty_world()
    tracks = [track(0, BoxColor.RED, -0.6, world), track(1, BoxColor.BLUE, -0.1, world),     # Robot 1: blue leaves first
              track(2, BoxColor.GREEN, 0.5, world), track(3, BoxColor.GRAY, 0.8, world)]      # Robot 2: gray leaves first
    by_robot = {a.robot_id: a for a in allocate_all(world, tracks)}
    assert (by_robot[1].track_id, by_robot[1].target_region) == (1, "region_C")
    assert (by_robot[2].track_id, by_robot[2].target_region) == (3, "bin_gray")
    assert by_robot[1].pick_time == world.time + LEAD_TIME
    assert by_robot[1].pick_position[0] == pytest.approx(-0.1)


def test_green_goes_to_whichever_robot_can_take_it():
    world = empty_world()
    upstream, downstream = track(0, BoxColor.GREEN, -0.3, world), track(1, BoxColor.GREEN, 0.6, world)
    assert [(a.robot_id, a.target_region) for a in allocate_all(world, [upstream])] == [(1, "region_B")]
    assert [(a.robot_id, a.target_region) for a in allocate_all(world, [downstream])] == [(2, "region_B")]
    assert allocate_all(world, [downstream], roles=FIXED_ROLES) == []       # with fixed roles Robot 2 leaves green alone
    assert allocate_all(world, [upstream], free=[2]) == []                  # Robot 1 is busy: the box travels on


def test_a_region_receives_one_box_at_a_time():
    world = empty_world()
    greens = [track(0, BoxColor.GREEN, -0.3, world), track(1, BoxColor.GREEN, 0.6, world)]
    assert len(allocate_all(world, greens)) == 1                            # both robots free, but only one region B
    assert allocate(world, greens, {0: 0.0, 1: 0.0}, {}, SHARED_ROLES, [2], busy_regions={"region_B"}) == []


def test_tracks_that_must_not_be_picked_are_skipped():
    world = empty_world()
    red = track(0, BoxColor.RED, -0.4, world)
    assert len(allocate_all(world, [red])) == 1
    assert allocate_all(world, [red], blocked={0: world.time + 1.0}) == []                    # retry not due yet
    assert allocate(world, [red], {0: world.time - 0.5}, {}, SHARED_ROLES, [1, 2]) == []      # followed too briefly
    red.timestamp = world.time - 1.0
    assert allocate_all(world, [red]) == []                                                   # not seen recently
    assert allocate_all(world, [track(1, BoxColor.RED, 0.6, world)]) == []                    # past Robot 1, not Robot 2's color
    assert allocate_all(world, [track(2, BoxColor.GRAY, 0.5, world, y=0.9)]) == []            # out of reach


def test_no_box_is_assigned_to_both_robots_and_no_robot_gets_two_boxes():
    world = empty_world()
    rng = np.random.default_rng(0)
    n_assignments = 0
    for _ in range(30):
        tracks = [track(i, BoxColor(rng.choice([c.value for c in BoxColor])), rng.uniform(-1.0, 1.0), world,
                        y=rng.uniform(0.12, 0.28)) for i in range(8)]
        assignments = allocate_all(world, tracks)
        assert len({a.track_id for a in assignments}) == len(assignments) == len({a.robot_id for a in assignments})
        for a in assignments:
            color = tracks[a.track_id].color
            assert color in SHARED_ROLES[a.robot_id] and a.target_region == DEFAULT_TARGET[color]
        n_assignments += len(assignments)
    assert n_assignments > 40


def test_shared_roles_in_the_sorting_loop():
    world = World(ConveyorConfig(speed=0.1, spacing=0.4, seed=1))           # a box every 4 s: more than Robot 1 can sort
    stats = run_robots(world, GroundTruthDetector(world).detect, duration=90.0, roles=SHARED_ROLES)
    green_by_robot_2 = [e for e in stats.events if e.robot_id == 2 and e.color == BoxColor.GREEN]
    assert len(green_by_robot_2) >= 2 and all(e.correct and e.region == "region_B" for e in green_by_robot_2)
    assert stats.failed == 0 and stats.sorted_correctly == len(stats.events) >= 13
    fixed = World(ConveyorConfig(speed=0.1, spacing=0.4, seed=1))
    assert run_robots(fixed, GroundTruthDetector(fixed).detect, duration=90.0).sorted_correctly < stats.sorted_correctly
