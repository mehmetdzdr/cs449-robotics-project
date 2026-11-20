import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.planning.reach import is_reachable
from conveyor_sort.sim.regions import (BUFFER, BUFFER_ROBOTS, DEFAULT_TARGET, REGIONS, SLOTS, region_contains,
                                       slot_position)
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


def test_buffer_is_reachable_by_both_robots_and_not_a_target():
    C = load_scene()
    assert BUFFER in C.getFrameNames() and BUFFER not in REGIONS and set(BUFFER_ROBOTS) == {1, 2}
    for slot in range(len(SLOTS)):                  # every slot, at the height of a placed box
        p = np.append(slot_position(C, BUFFER, slot), 0.65)
        assert all(is_reachable(C, r, p) for r in (1, 2))
