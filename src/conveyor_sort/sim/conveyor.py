"""Conveyor belt: geometry read from the `belt` frame in assets/scene.g. Boxes travel along +x.

Two backends:
- kinematic (default): a box is a child frame of the belt, the belt moves it by writing its relative pose.
- physics (`physics=True`, used by `World(physics=True)`): boxes are free rigid bodies in BotOp's physics
  simulation. They must all exist before the simulation starts (a body added later crashes `bot.sync`), so a
  pool of `ConveyorConfig.pool` boxes waits on the ground out of view and a spawn moves the next one to the belt start.
  The belt drives every box resting on its surface by `speed * dt` per step (`bot.cheat_setFramePose`); a box
  lifted off the belt or held by a gripper (`hold`) is never written to, so it moves by physics only; a box
  behind it in the same lane jams (stops 2 mm before it) instead of being moved into it. A box whose centre
  passes the belt end is counted as missed and taken out (some tip over the edge, some hang on it).
"""
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
    pool: int = 60                   # physics backend: boxes created before the simulation starts (max. spawns)


PAIR_GAP = 0.08   # distance between the two boxes of a blocked pair [m]: the gray one is in the way of a grasp
BOX_MASS = 0.1    # physics backend [kg]
ON_BELT_TOL = 0.01      # physics backend: a box is on the belt if its centre is within this of the resting height [m]
JAM_GAP = 0.002         # physics backend: a box jammed behind one that is not driven stops this far from it [m]


def _storage_position(index: int) -> list[float]:
    """Physics backend: unused and removed boxes rest on the ground (PhysX's ground plane at z = 0, which pushes
    anything below it back up), behind the robots and outside the top camera's view (y about -0.65 to 0.95)."""
    return [-1.0 + 0.1 * (index % 20), -2.5 - 0.1 * (index // 20), BOX_SIZE / 2]


class Conveyor:
    """Owns the boxes in the scene. A box is 'on the belt' while its parent frame is the belt."""

    def __init__(self, C: ry.Config, config: ConveyorConfig | None = None, physics: bool = False):
        self.C = C
        self.config = config or ConveyorConfig()
        self.physics = physics
        self.bot: ry.BotOp | None = None   # physics backend: set by World once the simulation runs
        self.held: set[str] = set()        # physics backend: boxes a gripper has closed on (see hold)
        if physics:
            for i in range(self.config.pool):
                C.addFrame(f"box_{i}").setShape(ry.ST.ssBox, [BOX_SIZE, BOX_SIZE, BOX_SIZE, 0.005]).setPosition(
                    _storage_position(i)).setMass(BOX_MASS).setContact(1).setColor(BOX_RGB[BoxColor.GRAY])
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
        if self.physics and self._next_id >= self.config.pool:
            raise RuntimeError(f"all {self.config.pool} boxes of the physics pool are used: raise ConveyorConfig.pool")
        self._next_id += 1
        jitter = self._rng.uniform(-self.config.lateral_jitter, self.config.lateral_jitter)
        y = jitter if y is None else y
        belt = self.C.getFrame(BELT)
        if self.physics:
            g = self.geometry
            position = [g.x_start + BOX_SIZE / 2 if x is None else x, g.y + y, g.z_top + BOX_SIZE / 2]
            self.C.getFrame(name).setColor(BOX_RGB[color])
            self._set_pose(name, position, [1.0, 0, 0, 0])
            self.colors[name] = color
            return name
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
        if not self.physics:
            return [name for name in self.colors if self.C.getFrame(name).getParent().name == BELT]
        g = self.geometry
        carried = []
        for name in self.colors:
            x, y, z = self.C.getFrame(name).getPosition()
            if (name not in self.held and g.x_start <= x <= g.x_end and abs(y - g.y) <= g.width / 2
                    and abs(z - g.z_top - BOX_SIZE / 2) < ON_BELT_TOL):
                carried.append(name)
        return carried

    def hold(self, box: str) -> None:
        """Physics backend: a gripper is about to close on `box`; the belt stops driving it (no pose writes on a
        box in a gripper). The box then rests on the belt surface until it is lifted."""
        self.held.add(box)

    def unhold(self, box: str) -> None:
        """Physics backend: the grasp failed or the box was released; the belt drives it again if it is on it."""
        self.held.discard(box)

    def _set_pose(self, name: str, position, quaternion) -> None:
        """Physics backend: move a box in the simulation (and in C, which the next sync would update anyway)."""
        frame = self.C.getFrame(name)
        frame.setPosition(position).setQuaternion(quaternion)
        if self.bot is not None:      # before the start, the simulation takes the poses from C
            self.bot.cheat_setFramePose(name, np.concatenate([position, quaternion]))

    def attach(self, box: str, parent: str) -> None:
        """Re-parent a box (to a gripper when grasped, to a region when placed), keeping its world pose."""
        self.C.getFrame(box).setParent(self.C.getFrame(parent), True)

    def remove(self, box: str) -> None:
        """Take a box out of the simulation.

        The frame is not deleted: after `delFrame`, RAI's collision engine keeps stale geometry and KOMO
        returns wrong results or crashes. The box is parked below the floor instead, each at its own spot.
        """
        if self.physics:
            self._set_pose(box, _storage_position(int(box.split("_")[1])), [1.0, 0, 0, 0])
            del self.colors[box]
            return
        frame = self.C.getFrame(box)
        frame.unLink()
        frame.setPosition([0.1 * int(box.split("_")[1]), 0.0, -1.0])
        del self.colors[box]

    def step(self, dt: float) -> None:
        """Advance the belt by dt: move boxes, drop the ones past the end, spawn new ones by spacing."""
        advance = self.config.speed * dt
        if self.physics:
            for name in list(self.colors):    # centre past the belt end: falls off (or hangs on the edge, pushed
                x, _, z = self.C.getFrame(name).getPosition()          # into by the box behind it): missed
                if x > self.geometry.x_end and z < self.geometry.z_top + BOX_SIZE and name not in self.held:
                    self.remove(name)
                    self.missed += 1
            # front box first: the simulation keeps stepping between two pose writes, and a box moved into
            # the one in front of it would push it away
            driven = self.on_belt()
            g = self.geometry
            stops = [self.C.getFrame(n).getPosition() for n in self.colors if n not in driven]
            stops = [p for p in stops if abs(p[2] - g.z_top - BOX_SIZE / 2) < BOX_SIZE]   # at belt height
            for name in sorted(driven, key=lambda n: -self.C.getFrame(n).getPosition()[0]):
                frame = self.C.getFrame(name)
                position = frame.getPosition()
                x = position[0] + advance
                for p in stops:   # jam behind a box that is not driven (held by a gripper, or tipped over)
                    if abs(p[1] - position[1]) < BOX_SIZE and position[0] < p[0] < x + BOX_SIZE:
                        x = max(position[0], min(x, p[0] - BOX_SIZE - JAM_GAP))
                self._set_pose(name, [x, position[1], position[2]], frame.getQuaternion())
        else:
            for name in self.on_belt():
                frame = self.C.getFrame(name)
                if frame.getPosition()[0] + advance > self.geometry.x_end:
                    self.remove(name)
                    self.missed += 1
                else:
                    frame.setRelativePosition(frame.getRelativePosition() + [advance, 0, 0])

        # a box entering during this step has already travelled the overshoot (matters for the variable
        # steps of the physics clock; with fixed steps the overshoot is zero)
        entry = self.geometry.x_start + BOX_SIZE / 2
        self._since_spawn += advance
        if self._since_spawn >= self.config.spacing:
            self._since_spawn -= self.config.spacing
            if self.config.max_boxes is None or self._next_id < self.config.max_boxes:
                colored = [c for c, weight in self.config.color_weights.items() if c != BoxColor.GRAY and weight > 0]
                if colored and self.config.blocked_pairs and self._rng.random() < self.config.blocked_pairs:
                    gray = self.spawn_box(BoxColor.GRAY, x=entry + self._since_spawn)
                    self._follower = PAIR_GAP, self.C.getFrame(gray).getRelativePosition()[1]
                    self.pairs += 1
                else:
                    self.spawn_box(x=entry + self._since_spawn)
        if self._follower:
            travel, y = self._follower[0] - advance, self._follower[1]
            self._follower = (travel, y)
            if travel <= 1e-9:
                colored = [c for c, weight in self.config.color_weights.items() if c != BoxColor.GRAY and weight > 0]
                self.spawn_box(colored[self._rng.choice(len(colored))], x=entry - travel, y=y)
                self._follower = None
