"""Sorting loop, command mode and clear-and-sort on the physics backend (#96).

Physics counterparts of the scenario tests in test_sorting.py, test_command_mode.py and test_clear_and_sort.py.
Tolerances differ where noted: BotOp's simulation runs in real time, so two runs are not identical.
"""
import pytest

from conveyor_sort.command.mode import CommandMode
from conveyor_sort.coordination.allocation import FIXED_ROLES
from conveyor_sort.interfaces import BoxColor, Command
from conveyor_sort.perception.detectors import get_detector
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.regions import region_contains
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots, run_single_robot

pytestmark = pytest.mark.slow   # physics scenarios in real time: nightly / on demand in CI (#97)

COLORED = {BoxColor.RED: 1.0, BoxColor.GREEN: 1.0, BoxColor.BLUE: 1.0}


@pytest.fixture
def physics_world():
    worlds = []

    def make(**config) -> World:
        worlds.append(World(ConveyorConfig(**config), physics=True))
        return worlds[-1]

    yield make
    for world in worlds:
        world.close()


def placed_freely(world, box, region) -> bool:
    """The box rests in the region as a free body (no kinematic release)."""
    return region_contains(world.C, region, world.C.getFrame(box).getPosition()) and world.C.getFrame(box).getParent() is None


def test_single_robot_sorts_every_box_with_ground_truth(physics_world):
    world = physics_world(speed=0.1, spacing=1.2, color_weights=COLORED, seed=0)    # a colored box every 12 s
    stats = run_single_robot(world, get_detector("gt", world), duration=65.0)   # kinematic: 60 s; a physical pick and
    print(f"{stats.sorted_correctly} of {len(stats.events)} sorted, {stats.missed} missed, "
          f"failed stages {[e.failed_stage for e in stats.events if not e.success]}")
    assert len(stats.events) == 5 and stats.sorted_correctly == 5     # place takes about 2 s longer (squeeze, settling)
    assert stats.failed == 0 and stats.missed == 0


def test_single_robot_sorts_with_hsv_vision_on_botop_images(physics_world):
    world = physics_world(speed=0.1, spacing=1.2, color_weights=COLORED, seed=1)
    stats = run_single_robot(world, get_detector("hsv", world), duration=16.0)    # one box (kinematic: 13 s; the
    print(f"{stats.sorted_correctly} of {len(stats.events)} sorted")                # images arrive 2 per s)
    assert len(stats.events) == 1 and stats.sorted_correctly == 1


def test_two_robots_with_fixed_roles(physics_world):
    world = physics_world(speed=0.1, spacing=0.6, seed=0)                         # a box every 6 s, 1 in 4 gray
    stats = run_robots(world, get_detector("gt", world), 60.0, FIXED_ROLES)
    by_robot = {r: [e for e in stats.events if e.robot_id == r] for r in (1, 2)}
    print(f"robot 1: {sum(e.correct for e in by_robot[1])} of {len(by_robot[1])}, robot 2: "
          f"{sum(e.correct for e in by_robot[2])} of {len(by_robot[2])}, missed {stats.missed}")
    # kinematic: exactly 5 and 2, all correct; here one attempt may fail (real-time simulation, see #95)
    assert sum(e.correct for e in by_robot[1]) >= 4 and sum(e.correct for e in by_robot[2]) >= 2
    assert all(e.color == BoxColor.GRAY for e in by_robot[2])
    assert stats.arm_contacts == []


@pytest.mark.parametrize("text, color, region, robot", [
    ("pick a blue box and place it at B", BoxColor.BLUE, "region_B", None),
    ("pick a red box and place it at C", BoxColor.RED, "region_C", 1),
    ("put a green box into the bin", BoxColor.GREEN, "bin_gray", 2),
])
def test_command_is_carried_out(physics_world, text, color, region, robot):
    world = physics_world(speed=0.1, spacing=0.4, seed=0)
    result = CommandMode(world, get_detector("gt", world)).execute(text, timeout=90.0)
    print(f"'{text}': robot {result.robot_id}, {result.time:.1f} s, {result.failed_stage}")
    assert result.success and result.correct and result.command == Command(color, region)
    assert world.conveyor.colors[result.box] == color and placed_freely(world, result.box, region)
    assert robot is None or result.robot_id == robot


def test_several_commands_into_the_same_region_keep_every_box(physics_world):
    world = physics_world(speed=0.1, spacing=0.4, seed=1)
    mode = CommandMode(world, get_detector("gt", world))
    results = [mode.execute(text) for text in ("pick a red box and place it at B", "pick a blue box and place it at B")]
    print([(r.robot_id, r.failed_stage, round(r.time, 1)) for r in results])
    assert all(r.correct for r in results)
    assert all(placed_freely(world, r.box, "region_B") for r in results)     # kinematic: the first one is removed


def test_timeout_when_no_box_of_the_color_arrives(physics_world):
    world = physics_world(speed=0.1, spacing=0.4, color_weights={BoxColor.RED: 1.0}, seed=0)
    result = CommandMode(world, get_detector("gt", world)).execute("pick a blue box and place it at A", timeout=10.0)
    assert not result.success and result.failed_stage == "timeout" and result.robot_id is None
    assert result.time == pytest.approx(10.0, abs=2 * world.dt)               # steps are dt or a little longer


def test_blocked_pair_is_left_alone_by_clear_and_sort(physics_world):
    """Clear-and-sort is not available on physics (see plan_clear_and_sort): the pair travels on, no collision."""
    world = physics_world(speed=0.05, max_boxes=0)
    world.conveyor.spawn_box(BoxColor.RED, x=-0.9, y=0.0)
    world.conveyor.spawn_box(BoxColor.GRAY, x=-0.82, y=0.0)
    stats = run_robots(world, get_detector("gt", world), 45.0, FIXED_ROLES, clear_and_sort=True)
    assert stats.clear_and_sort == [] and stats.arm_contacts == []
    assert all(e.color == BoxColor.GRAY for e in stats.events)                # at most Robot 2 takes the gray box
