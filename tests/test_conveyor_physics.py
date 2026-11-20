"""Physical conveyor (#89): boxes are rigid bodies in BotOp's simulation, driven by the belt."""
import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.sim.conveyor import BOX_SIZE, ConveyorConfig
from conveyor_sort.sim.world import World

REST_Z = 0.62 + BOX_SIZE / 2


@pytest.fixture
def physics_world():
    worlds = []

    def make(speed: float, hyper_speed: float = 5.0, **kwargs) -> World:
        kwargs.setdefault("max_boxes", 0)                  # scripted boxes only
        worlds.append(World(ConveyorConfig(speed=speed, **kwargs), physics=True, hyper_speed=hyper_speed))
        return worlds[-1]

    yield make
    for world in worlds:
        world.close()


def position(world, box):
    return world.C.getFrame(box).getPosition()


@pytest.mark.parametrize("speed", [0.05, 0.1, 0.2, 0.3])
def test_box_speed_matches_belt_speed(physics_world, speed):
    world = physics_world(speed)
    box = world.conveyor.spawn_box(BoxColor.RED, x=-0.9, y=0.0)
    world.run(1.0)                                         # settle
    x0, t0 = position(world, box)[0], world.time
    world.run(5.0)
    measured = (position(world, box)[0] - x0) / (world.time - t0)
    print(f"belt {speed} m/s: box {measured:.4f} m/s")
    assert measured == pytest.approx(speed, rel=0.05)
    assert position(world, box)[2] == pytest.approx(REST_Z, abs=0.005)


@pytest.mark.parametrize("y", [-0.12, 0.0, 0.12])
def test_no_lateral_drift_over_the_belt(physics_world, y):
    world = physics_world(0.3)
    box = world.conveyor.spawn_box(BoxColor.BLUE, y=y)       # enters at the belt start
    world.run(0.5)
    start = position(world, box).copy()
    world.run(6.0)                                         # 1.8 m of the 2.2 m belt
    end = position(world, box)
    print(f"y offset {y}: drift {abs(end[1] - start[1]) * 1000:.1f} mm over {end[0] - start[0]:.2f} m")
    assert end[0] - start[0] > 1.7 and abs(end[1] - start[1]) < 0.01


def test_touching_boxes_stay_in_contact(physics_world):
    world = physics_world(0.2)
    a = world.conveyor.spawn_box(BoxColor.RED, x=-0.8, y=0.0)
    b = world.conveyor.spawn_box(BoxColor.GREEN, x=-0.8 + BOX_SIZE, y=0.0)
    world.run(5.0)
    gap = position(world, b)[0] - position(world, a)[0]
    print(f"touching boxes after 1 m: centre distance {gap * 1000:.1f} mm")
    assert gap == pytest.approx(BOX_SIZE, abs=0.003)                     # not pushed apart
    assert position(world, a)[2] == pytest.approx(REST_Z, abs=0.005)    # not stacked
    assert position(world, b)[2] == pytest.approx(REST_Z, abs=0.005)


def test_lifted_box_is_not_driven(physics_world):
    world = physics_world(0.2)
    box = world.conveyor.spawn_box(BoxColor.RED, x=-0.5, y=0.0)
    world.run(0.5)
    assert box in world.conveyor.on_belt()
    p = position(world, box).copy()
    world.conveyor._set_pose(box, p + [0, 0, 0.02], [1.0, 0, 0, 0])       # 2 cm above the belt
    assert box not in world.conveyor.on_belt()
    world.conveyor.step(0.5)
    assert position(world, box)[0] == pytest.approx(p[0])                # no pose write along the belt


def test_box_behind_a_held_box_jams(physics_world):
    world = physics_world(0.2)
    held = world.conveyor.spawn_box(BoxColor.RED, x=-0.3, y=0.0)
    behind = world.conveyor.spawn_box(BoxColor.GREEN, x=-0.4, y=0.0)
    other_lane = world.conveyor.spawn_box(BoxColor.BLUE, x=-0.4, y=0.1)
    world.run(0.3)
    world.conveyor.hold(held)                        # a gripper closes on it: the belt no longer drives it
    x_other, t0 = position(world, other_lane)[0], world.time
    world.run(2.0)
    gap = position(world, held)[0] - position(world, behind)[0]
    print(f"jammed box: centre distance {gap * 1000:.1f} mm")
    assert gap == pytest.approx(BOX_SIZE, abs=0.005)                        # stopped behind it, not pushed into it
    other_speed = (position(world, other_lane)[0] - x_other) / (world.time - t0)
    assert other_speed == pytest.approx(0.2, rel=0.05)                       # other lane: not stopped
    world.conveyor.unhold(held)
    x_behind, t0 = position(world, behind)[0], world.time
    world.run(1.0)
    assert (position(world, behind)[0] - x_behind) / (world.time - t0) == pytest.approx(0.2, rel=0.05)   # moves on


def test_box_at_the_belt_end_falls_off_and_is_missed(physics_world):
    world = physics_world(0.3)
    box = world.conveyor.spawn_box(BoxColor.GREEN, x=0.8, y=0.0)
    world.run(0.3)
    assert world.conveyor.missed == 0 and box in world.conveyor.on_belt()
    world.run(3.0)                                                       # driven until its centre passes the end
    assert world.conveyor.missed == 1 and box not in world.conveyor.colors
    assert position(world, box)[1] < -2.0                                  # back in storage


def test_stream_spawns_from_the_pool(physics_world):
    world = physics_world(0.3, spacing=0.3, max_boxes=None, pool=3)
    world.run(2.5)                                                         # spawns at 0, 1, 2 s
    assert world.conveyor.spawned == 3 and len(world.conveyor.on_belt()) == 3
    with pytest.raises(RuntimeError, match="pool"):
        world.run(1.0)
    xs = sorted(position(world, b)[0] for b in world.conveyor.on_belt())
    assert np.diff(xs) == pytest.approx([0.3, 0.3], abs=0.02)
