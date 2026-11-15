"""Conveyor belt: geometry read from the `belt` frame in assets/scene.g. Boxes travel along +x."""
from dataclasses import dataclass, field

import numpy as np
import robotic as ry

from conveyor_sort.interfaces import BoxColor

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


BOX_SIZE = 0.06  # cube edge [m]; fits the Panda gripper (max opening 0.08 m)
BOX_RGB = {
    BoxColor.RED: [1.0, 0.0, 0.0],
    BoxColor.GREEN: [0.0, 0.8, 0.0],
    BoxColor.BLUE: [0.0, 0.2, 1.0],
    BoxColor.GRAY: [0.55, 0.55, 0.55],
}


@dataclass
class ConveyorConfig:
    speed: float = 0.1               # belt speed [m/s]
    spacing: float = 0.3             # distance between consecutive boxes along the belt [m]
    lateral_jitter: float = 0.08     # boxes enter at belt centre +- this offset in y [m]
    color_weights: dict[BoxColor, float] = field(default_factory=lambda: {c: 1.0 for c in BoxColor})
    seed: int = 0
    max_boxes: int | None = None     # stop spawning after this many boxes (None: endless)
    blocked_pairs: float = 0.0       # probability that a spawn is a gray box with a colored box PAIR_GAP behind it


PAIR_GAP = 0.08   # distance between the two boxes of a blocked pair [m]: the gray one is in the way of a grasp


class Conveyor:
    """Owns the boxes in the scene. A box is 'on the belt' while its parent frame is the belt."""

    def __init__(self, C: ry.Config, config: ConveyorConfig | None = None):
        self.C = C
        self.config = config or ConveyorConfig()
        self.geometry = belt_geometry(C)
        self.colors: dict[str, BoxColor] = {}   # box frame name -> color, for every box in the scene
        self._rng = np.random.default_rng(self.config.seed)
        self._next_id = 0
        self._since_spawn = self.config.spacing   # belt travel since the last spawn; first box enters at once
        self.missed = 0                           # boxes that reached the belt end unpicked
        self.pairs = 0                            # blocked pairs spawned so far
        self._follower: tuple[float, float] | None = None   # (belt travel left, y) until the colored box of a pair enters

    @property
    def spawned(self) -> int:
        """Number of boxes that have entered the belt so far."""
        return self._next_id

    def spawn_box(self, color: BoxColor | None = None, x: float | None = None, y: float | None = None) -> str:
        """Add a box at the belt start; random color (by color_weights) if none is given.

        For scripted scenes: `x` (world frame) puts it elsewhere along the belt, `y` is the offset from the centre line.
        """
        if color is None:
            colors = list(self.config.color_weights)
            weights = np.array([self.config.color_weights[c] for c in colors], dtype=float)
            color = colors[self._rng.choice(len(colors), p=weights / weights.sum())]
        name = f"box_{self._next_id}"
        self._next_id += 1
        jitter = self._rng.uniform(-self.config.lateral_jitter, self.config.lateral_jitter)
        y = jitter if y is None else y
        belt = self.C.getFrame(BELT)
        frame = self.C.addFrame(name, BELT)
        frame.setShape(ry.ST.ssBox, [BOX_SIZE, BOX_SIZE, BOX_SIZE, 0.005])
        frame.setColor(BOX_RGB[color])
        frame.setContact(1)
        x_rel = -belt.getSize()[0] / 2 + BOX_SIZE / 2 if x is None else x - belt.getPosition()[0]
        frame.setRelativePosition([x_rel, y, belt.getSize()[2] / 2 + BOX_SIZE / 2])
        self.colors[name] = color
        return name

    def on_belt(self) -> list[str]:
        """Names of boxes currently carried by the belt (not grasped, not placed)."""
        return [name for name in self.colors if self.C.getFrame(name).getParent().name == BELT]

    def attach(self, box: str, parent: str) -> None:
        """Re-parent a box (to a gripper when grasped, to a region when placed), keeping its world pose."""
        self.C.getFrame(box).setParent(self.C.getFrame(parent), True)

    def remove(self, box: str) -> None:
        """Take a box out of the simulation.

        The frame is not deleted: after `delFrame`, RAI's collision engine keeps stale geometry and KOMO
        returns wrong results or crashes. The box is parked below the floor instead, each at its own spot.
        """
        frame = self.C.getFrame(box)
        frame.unLink()
        frame.setPosition([0.1 * int(box.split("_")[1]), 0.0, -1.0])
        del self.colors[box]

    def step(self, dt: float) -> None:
        """Advance the belt by dt: move boxes, drop the ones past the end, spawn new ones by spacing."""
        advance = self.config.speed * dt
        for name in self.on_belt():
            frame = self.C.getFrame(name)
            if frame.getPosition()[0] + advance > self.geometry.x_end:
                self.remove(name)
                self.missed += 1
            else:
                frame.setRelativePosition(frame.getRelativePosition() + [advance, 0, 0])

        self._since_spawn += advance
        if self._since_spawn >= self.config.spacing:
            self._since_spawn -= self.config.spacing
            if self.config.max_boxes is None or self._next_id < self.config.max_boxes:
                colored = [c for c, weight in self.config.color_weights.items() if c != BoxColor.GRAY and weight > 0]
                if colored and self.config.blocked_pairs and self._rng.random() < self.config.blocked_pairs:
                    gray = self.spawn_box(BoxColor.GRAY)
                    self._follower = PAIR_GAP, self.C.getFrame(gray).getRelativePosition()[1]
                    self.pairs += 1
                else:
                    self.spawn_box()
        if self._follower:
            travel, y = self._follower[0] - advance, self._follower[1]
            self._follower = (travel, y)
            if travel <= 1e-9:
                colored = [c for c, weight in self.config.color_weights.items() if c != BoxColor.GRAY and weight > 0]
                self.spawn_box(colored[self._rng.choice(len(colored))], y=y)
                self._follower = None
