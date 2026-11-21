import numpy as np
import pytest

from conveyor_sort.planning.reach import is_reachable
from conveyor_sort.interfaces import BoxColor
from conveyor_sort.sim.conveyor import BOX_SIZE, Conveyor, ConveyorConfig, belt_geometry
from conveyor_sort.sim.scene import load_scene


def test_belt_geometry():
    belt = belt_geometry(load_scene())
    assert belt.x_start == pytest.approx(-1.1)
    assert belt.x_end == pytest.approx(1.1)
    assert belt.z_top == pytest.approx(0.62)


def test_scene_with_belt_is_collision_free():
    C = load_scene()
    C.computeCollisions()
    assert C.getCollisions(0.0) == []


@pytest.mark.parametrize("x, robot1, robot2", [
    (-0.8, False, False),  # upstream: the robots stand at the middle of the belt, on opposite sides (#154)
    (-0.4, True, True),    # both robots reach x in [-0.5, 0.5] across the whole belt
    (0.0, True, True),
    (0.4, True, True),
    (0.8, False, False),
])
def test_belt_reachability(x, robot1, robot2):
    C = load_scene()
    belt = belt_geometry(C)
    for y in (belt.y - 0.1, belt.y, belt.y + 0.1):
        p = np.array([x, y, belt.z_top + 0.03])
        assert is_reachable(C, 1, p) == robot1
        assert is_reachable(C, 2, p) == robot2


def test_spawn_box_places_box_at_belt_start():
    C = load_scene()
    conveyor = Conveyor(C)
    name = conveyor.spawn_box(BoxColor.BLUE)
    belt = conveyor.geometry
    x, y, z = C.getFrame(name).getPosition()
    assert x == pytest.approx(belt.x_start + BOX_SIZE / 2)
    assert abs(y - belt.y) <= conveyor.config.lateral_jitter
    assert z == pytest.approx(belt.z_top + BOX_SIZE / 2)
    assert conveyor.colors[name] == BoxColor.BLUE
    assert conveyor.on_belt() == [name]
    C.computeCollisions()
    assert C.getCollisions(0.0) == []


def test_spawn_names_are_unique_and_colors_reproducible():
    def colors(seed):
        conveyor = Conveyor(load_scene(), ConveyorConfig(seed=seed))
        names = [conveyor.spawn_box() for _ in range(20)]
        assert len(set(names)) == 20
        return [conveyor.colors[n] for n in names]

    assert colors(1) == colors(1)
    assert colors(1) != colors(2)


def test_color_weights_control_distribution():
    config = ConveyorConfig(color_weights={BoxColor.RED: 1.0, BoxColor.GRAY: 0.0})
    conveyor = Conveyor(load_scene(), config)
    assert {conveyor.colors[conveyor.spawn_box()] for _ in range(10)} == {BoxColor.RED}


def test_step_moves_boxes_with_belt_speed():
    C = load_scene()
    conveyor = Conveyor(C, ConveyorConfig(speed=0.2, spacing=100.0))
    conveyor.step(0.0)                      # first box enters immediately
    (name,) = conveyor.on_belt()
    x0 = C.getFrame(name).getPosition()[0]
    for _ in range(10):
        conveyor.step(0.05)
    assert C.getFrame(name).getPosition()[0] == pytest.approx(x0 + 0.2 * 0.5)


def test_boxes_spawn_with_configured_spacing():
    C = load_scene()
    conveyor = Conveyor(C, ConveyorConfig(speed=0.1, spacing=0.25))
    for _ in range(int(10 / 0.05)):         # 10 s -> 1.0 m of belt travel
        conveyor.step(0.05)
    xs = sorted(C.getFrame(n).getPosition()[0] for n in conveyor.on_belt())
    assert len(xs) == 5
    assert np.diff(xs) == pytest.approx(0.25, abs=0.006)


def test_box_leaving_belt_end_is_removed_and_counted():
    C = load_scene()
    conveyor = Conveyor(C, ConveyorConfig(speed=1.0, spacing=100.0))
    conveyor.step(0.0)
    for _ in range(50):                     # 2.5 m of travel > belt length
        conveyor.step(0.05)
    assert conveyor.on_belt() == []
    assert conveyor.missed == 1
    assert conveyor.colors == {}
    assert C.getFrame("box_0").getPosition()[2] < 0          # parked below the floor, not deleted
