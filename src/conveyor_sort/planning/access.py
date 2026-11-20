"""Grasp accessibility (#99): which grasp axes of a target box are free for the fingers, and which boxes block each.

The Panda's fingers close on two opposite faces of the box. "along": along the belt (faces at -x / +x, the default
of plan_moving_pick), "across": across it (faces at -y / +y, `across_belt=True`). A neighbour blocks an axis if it
is closer than BLOCK[axis] along that axis and closer than BLOCK_Y across it: an open finger then hits it, or lands
on it. The limits differ (measured with a box 6 cm left and right, or in front and behind, physics, Robot 1):

| neighbours (centre distance) | along, stopped / 0.1 m/s | across, stopped / 0.1 m/s |
| touching (6.1 cm)            | grasp fails / fails      | not plannable / fails      |
| 3 cm gap (9 cm)              | held / grasp fails       | held / held                |

Along the belt the box is 5 to 15 mm off at pick time on a moving belt (#91), so a 3 cm gap is not enough there
(allocation.BLOCK_X: hit at 0.10 m, free at 0.12 m); across it the position is precise.
"""
from dataclasses import dataclass

import numpy as np

from conveyor_sort.coordination.allocation import BLOCK_X, BLOCK_Y
from conveyor_sort.interfaces import BoxColor

AXES = ("along", "across")
BLOCK = {"along": BLOCK_X, "across": 0.09}   # a neighbour closer than this along the axis blocks it [m]


@dataclass
class GraspAccess:
    blockers: dict[str, list[int]]   # axis -> indices (into `boxes`) of the boxes that block it

    @property
    def free(self) -> list[str]:
        """The axes no box blocks, in the order of AXES."""
        return [axis for axis in AXES if not self.blockers[axis]]


def grasp_access(target: np.ndarray, boxes: list[tuple[BoxColor, np.ndarray]]) -> GraspAccess:
    """Free grasp axes of the box at `target` (world xy, or xyz) among `boxes` ((color, position) of the other
    boxes, any color: a blocker of another color than gray is reported like a gray one)."""
    blockers = {axis: [] for axis in AXES}
    for i, (_, position) in enumerate(boxes):
        dx, dy = np.abs(np.asarray(position)[:2] - np.asarray(target)[:2]) + 1e-9   # exactly at the limit: free
        if dx < BLOCK["along"] and dy < BLOCK_Y:
            blockers["along"].append(i)
        if dy < BLOCK["across"] and dx < BLOCK_Y:
            blockers["across"].append(i)
    return GraspAccess(blockers)
