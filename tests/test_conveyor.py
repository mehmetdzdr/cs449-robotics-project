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
    (-0.6, True, False),   # only Robot 1
    (0.0, True, True),     # shared workspace in the middle of the belt
    (0.6, False, True),    # only Robot 2
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
