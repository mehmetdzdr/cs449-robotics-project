"""Covered boxes (#177, opt-in): a hollow, open-bottom gray cover over a colored box on the belt.

A cover is one rigid body: a knob on top (the root frame, the same 6 cm cube as a box, so the gripper squeezes it
exactly like a box, with the same finger reference and holding check), a top plate and 4 walls. The walls enclose
the box with 1.5 cm on every side and reach 2 cm above it. Like the boxes (#89), the covers must exist before
BotOp starts: `CoveredConveyor` creates a pool of them, parked on the ground out of view, and puts one over a box.

The mode is switched on with `World(..., covers=CoverConfig(...))`, which builds a `CoveredConveyor` instead of a
`Conveyor`. Without it the scene has no cover frames and nothing here runs.

`CoveredConveyor` on top of `Conveyor`:
- a box under a cover that is down (its walls reach below the box top) is not "on the belt": no detector reports it
  (the camera cannot see it) and the belt does not drive it on its own;
- the belt drives a cover resting on it together with the box under it, in steps of at most MAX_SUBSTEP: the box has
  1.5 cm to the walls, and a long step (after a KOMO plan, up to 0.6 s) written in one go would put it into a wall;
- a held cover (`hold`, by a gripper) is not driven, nor is the box under it; once the cover is lifted the box is on
  the belt again and driven as before;
- `cover_spot` on Robot 2's side of the belt is where covers are put aside; `clear_spot` takes them away (as an
  operator would) before the next one is put there.
"""
from dataclasses import dataclass

import numpy as np
import robotic as ry

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.sim.conveyor import BOX_SIZE, ON_BELT_TOL, Conveyor, ConveyorConfig

COVER_WIDTH = 0.11       # outside, x and y [m]: inside 0.09, 1.5 cm around a 6 cm box
COVER_WALL = 0.01        # wall and plate thickness [m] (`box` shapes: thin ssBox shapes fail to load)
COVER_HEIGHT = 0.08      # inside height [m]: 2 cm above a 6 cm box
COVER_KNOB = BOX_SIZE    # knob edge [m]: grasped like a box
COVER_REST = COVER_KNOB / 2 + COVER_WALL + COVER_HEIGHT   # root (knob centre) above the surface the cover stands on [m]
COVER_MASS = 0.15        # [kg]; the root carries it, so the centre of mass is at the knob (no torque in the grasp)
COVER_RGB = [0.6, 0.6, 0.6]
COVER_INSIDE = COVER_WIDTH / 2 - COVER_WALL   # half the inside width [m]
COVER_MAX_OFFSET = 0.15 - COVER_WIDTH / 2     # a box is covered only this close to the belt centre line [m]
COVER_CLEARANCE = 0.01   # ... and with no other box or cover this close to the cover [m]
MAX_SUBSTEP = 0.005      # the belt moves a cover and its box in steps of at most this [m]
MAX_TILT = 10.0          # a cover tilted more is not standing on the belt (not driven) [deg]

COVER_SPOT = "cover_spot"                     # where Robot 2 puts the covers: its side of the belt, upstream
SPOT_POSITION = [-0.4, 0.85, 0.601]           # on the table extension (top at 0.6), like bin_gray mirrored in x
SPOT_SIZE = [0.2, 0.2, 0.002]


@dataclass
class CoverConfig:
    pool: int = 4         # covers created before the simulation starts (put-aside ones are reused)
    share: float = 0.0    # streams: share of colored boxes that enter under a cover (if one fits and is free)
    seed: int = 0         # for the cover draws, independent of the box stream (same boxes as without covers)


def _storage_position(index: int) -> list[float]:
    """Unused and put-aside covers stand on the ground (PhysX's ground plane at z = 0) behind Robot 1, out of view,
    beyond the parked boxes (y = -2.5 and further, see conveyor._storage_position)."""
    return [-1.0 + 0.2 * index, -3.5, COVER_REST]


def add_cover(C: ry.Config, name: str, position, physics: bool) -> None:
    """The cover as frames: `name` is the knob (root, with the mass on the physics backend), the plate and the walls
    are its children (one rigid body)."""
    root = C.addFrame(name).setShape(ry.ST.ssBox, [COVER_KNOB, COVER_KNOB, COVER_KNOB, 0.005]).setPosition(position)
    root.setColor(COVER_RGB).setContact(1)
    plate_z = -COVER_KNOB / 2 - COVER_WALL / 2
    wall_z = -COVER_KNOB / 2 - COVER_WALL - COVER_HEIGHT / 2
    offset = COVER_WIDTH / 2 - COVER_WALL / 2
    parts = {"plate": ([COVER_WIDTH, COVER_WIDTH, COVER_WALL], [0, 0, plate_z]),
             "wall_x0": ([COVER_WALL, COVER_WIDTH, COVER_HEIGHT], [-offset, 0, wall_z]),
             "wall_x1": ([COVER_WALL, COVER_WIDTH, COVER_HEIGHT], [offset, 0, wall_z]),
             "wall_y0": ([COVER_WIDTH - 2 * COVER_WALL, COVER_WALL, COVER_HEIGHT], [0, -offset, wall_z]),
             "wall_y1": ([COVER_WIDTH - 2 * COVER_WALL, COVER_WALL, COVER_HEIGHT], [0, offset, wall_z])}
    for part, (size, relative) in parts.items():
        frame = C.addFrame(f"{name}_{part}", name).setShape(ry.ST.box, size).setRelativePosition(relative)
        frame.setColor(COVER_RGB).setContact(1)
    if physics:
        root.setMass(COVER_MASS)


def cover_frames(C: ry.Config, cover: str) -> list[str]:
    """The root and the part frames of a cover."""
    return [cover] + [n for n in C.getFrameNames() if n.startswith(f"{cover}_")]


def tilt(C: ry.Config, frame: str) -> float:
    """Tilt of a frame from upright [deg]."""
    return float(np.degrees(np.arccos(np.clip(C.getFrame(frame).getRotationMatrix()[2, 2], -1.0, 1.0))))


class CoveredConveyor(Conveyor):
    """`Conveyor` with covers (see the module docstring). `covers` are the cover frame names."""

    def __init__(self, C: ry.Config, config: ConveyorConfig | None, covers: CoverConfig, physics: bool = False):
        super().__init__(C, config, physics=physics)
        self.cover_config = covers
        self.covers = [f"cover_{i}" for i in range(covers.pool)]
        for i, name in enumerate(self.covers):
            add_cover(C, name, _storage_position(i), physics)
        C.addFrame(COVER_SPOT).setShape(ry.ST.box, SPOT_SIZE).setPosition(SPOT_POSITION).setColor([0.9, 0.9, 0.75])
        self._parked = list(self.covers)
        self._cover_rng = np.random.default_rng([covers.seed, 177])
        self.covers_missed = 0      # covers that reached the belt end (with their box)
        self.covers_cleared = 0     # covers taken away from the spot

    def cover(self, box: str) -> str:
        """Put a free cover over `box` (scripted scenes; streams: `CoverConfig.share`)."""
        if not self._parked:
            raise RuntimeError(f"all {len(self.covers)} covers are in use: raise CoverConfig.pool")
        cover = self._parked.pop(0)
        x, y, _ = self.C.getFrame(box).getPosition()
        self._set_pose(cover, [x, y, self.geometry.z_top + COVER_REST], [1.0, 0, 0, 0])
        return cover

    def _park(self, cover: str) -> None:
        self._set_pose(cover, _storage_position(self.covers.index(cover)), [1.0, 0, 0, 0])
        self.held.discard(cover)
        self._parked.append(cover)

    def _active(self) -> list[str]:
        return [c for c in self.covers if c not in self._parked]

    def covered(self) -> dict[str, str]:
        """box -> the cover over it: a box at belt height inside the walls of a cover that is down."""
        g, result = self.geometry, {}
        for cover in self._active():
            p = self.C.getFrame(cover).getPosition()
            if p[2] - COVER_REST > g.z_top + BOX_SIZE:            # lifted: the walls are above the box tops
                continue
            for box in self.colors:
                q = self.C.getFrame(box).getPosition()
                if np.all(np.abs(q[:2] - p[:2]) < COVER_INSIDE) and abs(q[2] - g.z_top - BOX_SIZE / 2) < ON_BELT_TOL:
                    result[box] = cover
        return result

    def on_belt(self) -> list[str]:
        """Boxes on the belt that are not under a cover."""
        covered = self.covered()
        return [b for b in super().on_belt() if b not in covered]

    def covers_on_belt(self) -> list[str]:
        """Covers standing upright on the belt surface, not held by a gripper."""
        g, result = self.geometry, []
        for cover in self._active():
            x, y, z = self.C.getFrame(cover).getPosition()
            if (cover not in self.held and g.x_start <= x <= g.x_end and abs(y - g.y) <= g.width / 2
                    and abs(z - g.z_top - COVER_REST) < ON_BELT_TOL and tilt(self.C, cover) < MAX_TILT):
                result.append(cover)
        return result

    def coverable(self, box: str) -> bool:
        """A cover fits over `box`: near the centre line, no other box or cover in the way, a cover free."""
        p = self.C.getFrame(box).getPosition()
        reach = COVER_WIDTH / 2 + COVER_CLEARANCE
        others = [self.C.getFrame(b).getPosition() for b in self.colors if b != box]
        return (bool(self._parked) and abs(p[1] - self.geometry.y) <= COVER_MAX_OFFSET
                and not any(np.all(np.abs(q[:2] - p[:2]) < reach + BOX_SIZE / 2) for q in others)
                and not any(np.all(np.abs(self.C.getFrame(c).getPosition()[:2] - p[:2]) < reach + COVER_WIDTH / 2)
                            for c in self._active()))

    def spawn_box(self, color: BoxColor | None = None, x: float | None = None, y: float | None = None) -> str:
        box = super().spawn_box(color, x, y)
        share = self.cover_config.share
        if share and self.colors[box] != BoxColor.GRAY and self._cover_rng.random() < share and self.coverable(box):
            self.cover(box)
        return box

    def clear_spot(self) -> int:
        """Take the covers standing on the spot away; returns how many."""
        spot = self.C.getFrame(COVER_SPOT).getPosition()
        on_spot = [c for c in self._active() if c not in self.held
                   and np.all(np.abs(self.C.getFrame(c).getPosition()[:2] - spot[:2]) < np.array(SPOT_SIZE[:2]) / 2)]
        for cover in on_spot:
            self._park(cover)
        self.covers_cleared += len(on_spot)
        return len(on_spot)

    def step(self, dt: float) -> None:
        """Move the covers standing on the belt and the boxes under them, then everything else (`Conveyor.step`)."""
        advance = self.config.speed * dt
        covered = self.covered()
        for cover in self.covers_on_belt():
            frame = self.C.getFrame(cover)
            if frame.getPosition()[0] + advance > self.geometry.x_end:    # off the belt end (its box: Conveyor.step)
                self._park(cover)
                self.covers_missed += 1
                continue
            moved = [cover] + [b for b, c in covered.items() if c == cover]   # the cover first, then its box
            if not self.physics:
                frame.setPosition(frame.getPosition() + [advance, 0, 0])
                for box in moved[1:]:
                    box_frame = self.C.getFrame(box)
                    box_frame.setRelativePosition(box_frame.getRelativePosition() + [advance, 0, 0])
                continue
            start = {name: self.C.getFrame(name).getPosition() for name in moved}
            n = max(1, int(np.ceil(advance / MAX_SUBSTEP)))
            for k in range(1, n + 1):
                for name in moved:
                    self._set_pose(name, start[name] + [advance * k / n, 0, 0], self.C.getFrame(name).getQuaternion())
        super().step(dt)
