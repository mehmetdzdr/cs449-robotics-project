"""Full regions are emptied before the next place (#142), physics backend."""
import pytest

from conveyor_sort.command.mode import CommandMode
from conveyor_sort.interfaces import BoxColor
from conveyor_sort.sim.conveyor import BOX_SIZE, ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.regions import SLOTS, free_slot, region_contains, slot_position
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots

pytestmark = pytest.mark.slow   # physics scenarios in real time: nightly / on demand in CI (#97)


def world_with_full_region_a() -> tuple[World, list[str]]:
    """A red box on the belt (0.05 m/s) and region A full: a gray box in each of its 6 slots."""
    world = World(ConveyorConfig(speed=0.05, max_boxes=0), physics=True)
    world.conveyor.spawn_box(BoxColor.RED, x=-0.9, y=0.0)
    full = []
    for slot in range(len(SLOTS)):
        box = world.conveyor.spawn_box(BoxColor.GRAY, x=0.9, y=0.0)          # from the pool, then into the slot
        world.conveyor._set_pose(box, [*slot_position(world.C, "region_A", slot), 0.601 + 0.001 + BOX_SIZE / 2],
                                 [1.0, 0, 0, 0])
        full.append(box)
    world.run(0.5)
    assert free_slot(world.C, "region_A", world.conveyor.colors) is None
    return world, full


def check_emptied(world, full, box):
    assert all(b not in world.conveyor.colors for b in full)                 # collected
    assert region_contains(world.C, "region_A", world.C.getFrame(box).getPosition())


def test_sorting_loop_empties_a_full_region():
    world, full = world_with_full_region_a()
    try:
        stats = run_robots(world, GroundTruthDetector(world).detect, 25.0, {1: (BoxColor.RED,)})
        print([(e.failed_stage, round(e.time, 1)) for e in stats.events], stats.regions_emptied)
        (event,) = stats.events
        assert event.correct and stats.regions_emptied == 1
        (red,) = [b for b, c in world.conveyor.colors.items() if c == BoxColor.RED]
        check_emptied(world, full, red)
    finally:
        world.close()


def test_command_mode_empties_a_full_region():
    world, full = world_with_full_region_a()
    try:
        result = CommandMode(world, GroundTruthDetector(world).detect).execute("pick a red box and place it at A")
        print(result.failed_stage, round(result.time, 1))
        assert result.correct
        check_emptied(world, full, result.box)
    finally:
        world.close()
