"""Covered boxes on the physics backend (#177): the cover is lifted off by Robot 2 and put aside, the box under it is
sorted by Robot 1, with no arm contact. Planning-only counterparts in test_covers.py."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conveyor_sort.coordination.allocation import FIXED_ROLES
from conveyor_sort.coordination.covers import Uncovering
from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.perception.covers import get_cover_detector
from conveyor_sort.perception.detectors import get_detector
from conveyor_sort.planning.cover_task import CoverTask
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.covers import COVER_SPOT, CoverConfig, tilt
from conveyor_sort.sim.regions import region_contains
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots

pytestmark = pytest.mark.slow   # physics scenarios in real time: nightly / on demand in CI (#97)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def physics_world():
    worlds = []

    def make(covers: CoverConfig | None = CoverConfig(), **config) -> World:
        worlds.append(World(ConveyorConfig(**config), physics=True, covers=covers))
        return worlds[-1]

    yield make
    for world in worlds:
        world.close()


def on_spot(world, cover) -> bool:
    spot = world.C.getFrame(COVER_SPOT).getPosition()
    return bool(np.linalg.norm(world.C.getFrame(cover).getPosition()[:2] - spot[:2]) < 0.05 and tilt(world.C, cover) < 10)


def placed_freely(world, box, region) -> bool:
    return region_contains(world.C, region, world.C.getFrame(box).getPosition()) and world.C.getFrame(box).getParent() is None


def no_box_held(world) -> bool:
    fingers = world.C.getJointState()[list(world.fingers.values())]
    return not world.conveyor.held and bool(np.all(fingers > 0.035))


def test_default_physics_world_has_no_covers(physics_world):
    world = physics_world(covers=None, max_boxes=0)
    assert not [n for n in world.C.getFrameNames() if n.startswith("cover")]


def test_cover_is_lifted_off_the_moving_belt_and_put_aside(physics_world):
    """Robot 2 alone: the cover comes off, stands upright on the spot; the box under it is on the belt again and
    moves on with it, upright."""
    world = physics_world(speed=0.05, max_boxes=0)
    box = world.conveyor.spawn_box(BoxColor.RED, x=-0.6, y=0.0)
    cover = world.conveyor.cover(box)
    world.run(0.5)
    assert world.conveyor.covered() == {box: cover} and world.conveyor.covers_on_belt() == [cover]
    track = Track(0, BoxColor.GRAY, world.C.getFrame(cover).getPosition().copy(), np.array([0.05, 0, 0]), world.time)
    pick_time = world.time + 3.0
    task = CoverTask(world, Assignment(2, 0, COVER_SPOT, pick_time, track.predict(pick_time)), track)
    while task.advance():
        world.step()
    x_after = world.C.getFrame(box).getPosition()[0]
    world.run(1.0)
    print(task.result, f"box tilt {tilt(world.C, box):.1f} deg, moved {world.C.getFrame(box).getPosition()[0] - x_after:.3f} m")
    assert task.result.success and task.result.cover == cover and task.result.lifted_at is not None
    assert on_spot(world, cover) and task.result.place_error < 0.03
    assert box in world.conveyor.on_belt() and tilt(world.C, box) < 10
    assert world.C.getFrame(box).getPosition()[0] - x_after == pytest.approx(0.05, abs=0.02)   # driven again
    assert no_box_held(world)


@pytest.mark.parametrize("detector", ["gt", "hsv"])
def test_covered_box_is_uncovered_and_sorted(physics_world, detector):
    world = physics_world(speed=0.05, max_boxes=0)
    red = world.conveyor.spawn_box(BoxColor.RED, x=-0.9, y=0.0)
    cover = world.conveyor.cover(red)
    uncovering = Uncovering(world, get_cover_detector(detector, world))
    stats = run_robots(world, get_detector(detector, world), 45.0, FIXED_ROLES, covers=uncovering)
    print([(round(e.time, 1), e.robot_id, e.color.name, e.region, e.failed_stage) for e in stats.events],
          uncovering.stats, f"min arm distance {stats.min_arm_distance:.3f} m")
    assert uncovering.stats.lifted == uncovering.stats.put_aside == uncovering.stats.handed_over == 1
    assert [(e.robot_id, e.color, e.region, e.success) for e in stats.events] == [(1, BoxColor.RED, "region_A", True)]
    assert placed_freely(world, red, "region_A") and on_spot(world, cover)
    assert stats.arm_contacts == [] and uncovering.stats.cover_contacts == []
    assert no_box_held(world)


def test_stream_with_covered_boxes(physics_world):
    """Fixed roles, half the colored boxes under a cover: every lifted cover is put aside, its box handed over to
    Robot 1; no arm contact, no contact of Robot 1 with a cover."""
    world = physics_world(CoverConfig(share=0.5), speed=0.05, spacing=0.6, seed=0)
    uncovering = Uncovering(world, get_cover_detector("gt", world))
    stats = run_robots(world, get_detector("gt", world), 90.0, FIXED_ROLES, covers=uncovering)
    print([(round(e.time, 1), e.robot_id, e.color.name, e.region, e.failed_stage) for e in stats.events],
          uncovering.stats, f"missed {stats.missed}, min arm distance {stats.min_arm_distance:.3f} m")
    s = uncovering.stats
    assert s.lifted >= 2 and s.put_aside == s.lifted and s.handed_over == s.lifted
    assert stats.sorted_correctly >= s.handed_over
    assert stats.arm_contacts == [] and s.cover_contacts == []
    assert no_box_held(world)


def test_demo_scene():
    result = subprocess.run([sys.executable, str(SCRIPTS / "run_autonomous.py"), "--scene", "covered"],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    out = result.stdout
    print(out)
    assert "covers: 1 lifted, 1 put aside, 1 boxes handed over to robot 1" in out
    assert re.search(r"sorted correctly: 1, failed: 0,", out) and "0 contacts of robot 1 with a cover" in out
    assert out.rstrip().endswith("0 contacts")
