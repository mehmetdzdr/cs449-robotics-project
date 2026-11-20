import numpy as np
import pytest

from conveyor_sort.command.parser import parse_command
from conveyor_sort.command.selection import select
from conveyor_sort.coordination.allocation import LEAD_TIME
from conveyor_sort.interfaces import BoxColor, Command, Track
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.regions import REGION_ROBOTS, REGIONS
from conveyor_sort.sim.world import World

SPEED = 0.1


@pytest.fixture(scope="module")
def world() -> World:
    world = World(ConveyorConfig(speed=SPEED, spacing=100.0))
    world.time = 10.0
    return world


def track(track_id: int, color: BoxColor, pick_x: float, world: World, y: float = 0.2) -> Track:
    """A box on the belt that will be at `pick_x` at the pick time."""
    return Track(track_id, color, np.array([pick_x - SPEED * LEAD_TIME, y, 0.65]), np.array([SPEED, 0, 0]), world.time)


def select_from(world, text, tracks, **kwargs):
    return select(world, parse_command(text), tracks, {t.track_id: 0.0 for t in tracks}, **kwargs)


def test_every_region_has_a_robot():
    assert set(REGION_ROBOTS) == set(REGIONS)


def test_box_of_the_commanded_color_closest_to_the_robot_is_chosen(world):
    tracks = [track(0, BoxColor.RED, -0.4, world), track(1, BoxColor.BLUE, -0.9, world),
              track(2, BoxColor.BLUE, -0.45, world), track(3, BoxColor.BLUE, 0.1, world)]
    assignment = select_from(world, "pick a blue box and place it at A", tracks)
    assert (assignment.robot_id, assignment.track_id, assignment.target_region) == (1, 2, "region_A")   # base at x = -0.4
    assert assignment.pick_time == world.time + LEAD_TIME
    assert assignment.pick_position == pytest.approx([-0.45, 0.2, 0.65])


def test_robot_is_chosen_by_the_target_region(world):
    red = [track(0, BoxColor.RED, -0.4, world), track(1, BoxColor.RED, 0.3, world)]
    assert select_from(world, "put a red box into the bin", red).robot_id == 2          # only Robot 2 reaches the bin
    assert select_from(world, "put a red box into the bin", red).track_id == 1          # (it picks from x = -0.15)
    assert select_from(world, "pick a red box and place it at C", red).track_id == 0    # only Robot 1 reaches C
    assert select_from(world, "pick a red box and place it at C", red[1:]) is None      # the box has passed Robot 1
    assert select_from(world, "pick a red box and place it at B", red).robot_id == 1    # B: Robot 1's side (#154)
    assert select_from(world, "pick a red box and place it at B", red, robots=(2,)) is None


def test_no_feasible_box(world):
    command = "pick a green box and place it at B"
    assert select_from(world, command, []) is None
    assert select_from(world, command, [track(0, BoxColor.RED, -0.4, world)]) is None              # wrong color
    assert select_from(world, command, [track(0, BoxColor.GREEN, 0.5, world, y=0.9)]) is None      # out of reach
    assert select_from(world, command, [track(0, BoxColor.GREEN, 1.0, world)]) is None             # too little time left
    green = track(0, BoxColor.GREEN, -0.4, world)
    assert select_from(world, command, [green], blocked={0: world.time + 1.0}) is None
    assert select(world, Command(BoxColor.GREEN, "region_B"), [green], {0: world.time - 0.2}) is None   # followed too briefly
    assert select_from(world, command, [green]).track_id == 0
