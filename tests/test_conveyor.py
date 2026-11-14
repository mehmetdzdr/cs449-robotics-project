import numpy as np
import pytest

from conveyor_sort.planning.reach import is_reachable
from conveyor_sort.sim.conveyor import belt_geometry
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
