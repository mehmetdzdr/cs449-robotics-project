"""Shared data types passed between modules.

Every module talks to the others only through these types, so each member can
develop against ground-truth data (from the simulator) before the upstream
module is ready.

Pipeline:  camera -> perception -> prediction -> coordination -> planning
                                    command (optional) --^
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class BoxColor(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    GRAY = "gray"  # unwanted boxes, removed by Robot 2


@dataclass
class Detection:
    """One box seen in one camera frame (output of perception)."""
    color: BoxColor
    position: np.ndarray        # (3,) world-frame position [m]
    confidence: float           # 1.0 for HSV / ground truth
    timestamp: float            # simulation time [s]


@dataclass
class Track:
    """A box followed over time (output of prediction)."""
    track_id: int
    color: BoxColor
    position: np.ndarray        # (3,) latest estimated position [m]
    velocity: np.ndarray        # (3,) estimated velocity [m/s]
    timestamp: float

    def predict(self, t: float) -> np.ndarray:
        """Position at time t under a constant-velocity assumption."""
        return self.position + self.velocity * (t - self.timestamp)


@dataclass
class Command:
    """High-level user instruction, e.g. 'pick a blue box and place it at B'."""
    color: BoxColor
    target_region: str          # name of a region frame in the RAI config


@dataclass
class Assignment:
    """Which robot handles which box, and where it goes (output of coordination)."""
    robot_id: int               # 1 = sorter, 2 = gray-box remover
    track_id: int
    target_region: str
    pick_time: float            # planned grasp time [s]
    pick_position: np.ndarray   # (3,) predicted box position at pick_time
