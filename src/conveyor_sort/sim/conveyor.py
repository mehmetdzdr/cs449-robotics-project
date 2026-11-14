"""Conveyor belt: geometry read from the `belt` frame in assets/scene.g. Boxes travel along +x."""
from dataclasses import dataclass

import robotic as ry

BELT = "belt"


@dataclass(frozen=True)
class BeltGeometry:
    x_start: float   # where boxes enter [m]
    x_end: float     # where boxes fall off [m]
    y: float         # centre line [m]
    width: float     # extent in y [m]
    z_top: float     # height of the belt surface [m]


def belt_geometry(C: ry.Config) -> BeltGeometry:
    frame = C.getFrame(BELT)
    x, y, z = frame.getPosition()
    length, width, height = frame.getSize()[:3]
    return BeltGeometry(x - length / 2, x + length / 2, y, width, z + height / 2)
