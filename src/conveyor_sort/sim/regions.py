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


def region_contains(C: ry.Config, region: str, position: np.ndarray) -> bool:
    """True if the xy of `position` (world frame) lies inside the region's footprint."""
    frame = C.getFrame(region)
    half = frame.getSize()[:2] / 2
    return bool(np.all(np.abs(position[:2] - frame.getPosition()[:2]) <= half))
