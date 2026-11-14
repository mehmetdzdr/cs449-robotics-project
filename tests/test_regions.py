import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.planning.reach import is_reachable
from conveyor_sort.sim.regions import DEFAULT_TARGET, REGIONS, region_contains
from conveyor_sort.sim.scene import load_scene


def test_regions_exist_and_every_color_has_a_target():
    names = load_scene().getFrameNames()
    assert all(region in names for region in REGIONS)
    assert set(DEFAULT_TARGET) == set(BoxColor)
    assert set(DEFAULT_TARGET.values()) <= set(REGIONS)


def test_region_contains():
    C = load_scene()
    centre = C.getFrame("region_A").getPosition()
    assert region_contains(C, "region_A", centre + [0.09, -0.09, 0.03])
    assert not region_contains(C, "region_A", centre + [0.11, 0.0, 0.03])
    assert not region_contains(C, "region_B", centre)


@pytest.mark.parametrize("region, robots", [
    ("region_A", {1}), ("region_B", {1, 2}), ("region_C", {1}), ("bin_gray", {2}),
])
def test_region_reachability(region, robots):
    C = load_scene()
    p = C.getFrame(region).getPosition() + np.array([0, 0, 0.05])
    assert {r for r in (1, 2) if is_reachable(C, r, p)} == robots
