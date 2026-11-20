"""Target regions: named frames in assets/scene.g where boxes are placed."""
import numpy as np
import robotic as ry

from conveyor_sort.interfaces import BoxColor

REGIONS = ["region_A", "region_B", "region_C", "bin_gray"]

# autonomous mode: where each color goes
DEFAULT_TARGET = {
    BoxColor.RED: "region_A",
    BoxColor.GREEN: "region_B",
    BoxColor.BLUE: "region_C",
    BoxColor.GRAY: "bin_gray",
}

# which robots can place a box there (robot ids as in interfaces.Assignment)
REGION_ROBOTS = {"region_A": (1,), "region_B": (1, 2), "region_C": (1,), "bin_gray": (2,)}


# Physics backend: a region holds up to len(SLOTS) boxes, one per slot (offsets from the region centre [m]).
# 3 x 2 in the 0.2 m x 0.2 m region: 7 cm apart in x, 10 cm in y; the planner turns the gripper so that the
# fingers do not hit the boxes already there (open fingers reach about 5 cm from the box centre).
SLOTS = [(dx, dy) for dy in (-0.05, 0.05) for dx in (-0.07, 0.0, 0.07)]
SLOT_TOLERANCE = 0.03   # a box centre this close to a slot centre (in x and in y) occupies it [m]


def slot_position(C: ry.Config, region: str, slot: int) -> np.ndarray:
    """World xy of a slot centre."""
    return C.getFrame(region).getPosition()[:2] + SLOTS[slot]


def free_slot(C: ry.Config, region: str, boxes) -> int | None:
    """The first slot of `region` that none of `boxes` occupies, or None if the region is full."""
    occupied = [C.getFrame(b).getPosition()[:2] for b in boxes]
    for slot in range(len(SLOTS)):
        centre = slot_position(C, region, slot)
        if not any(np.all(np.abs(p - centre) < SLOT_TOLERANCE) for p in occupied):
            return slot
    return None


def region_contains(C: ry.Config, region: str, position: np.ndarray) -> bool:
    """True if the xy of `position` (world frame) lies inside the region's footprint."""
    frame = C.getFrame(region)
    half = frame.getSize()[:2] / 2
    return bool(np.all(np.abs(position[:2] - frame.getPosition()[:2]) <= half))
