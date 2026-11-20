"""Scripted belt layouts for the challenging scenarios (#98): a colored target box with gray neighbours.

`make_scenario(name, seed, lateral, spacing)` returns the layout as data (colors and world positions), so it can be
analysed (e.g. grasp accessibility) without a simulation; `spawn_scenario(world, scenario)` puts it on the belt.
Directions are those of the belt: "front" is downstream (+x, the direction of travel), "left" is the far side of
the belt (+y, away from the robots), "right" the near side.
"""
from dataclasses import dataclass, field

import numpy as np

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.sim.conveyor import BOX_SIZE, belt_geometry
from conveyor_sort.sim.world import World

# target offset from the belt centre line [m]: the belt is 0.3 m wide, a box centre fits within +-0.12 m
LATERAL = {"centre": 0.0, "near_edge": -0.12, "far_edge": 0.12}
# centre-to-centre distance between the target and its neighbours [m]: touching (1 mm apart, so that they do not
# spawn interpenetrating), a 3 cm gap, 11 cm (allocation.BLOCK_X: from here on the fingers pass), 30 cm (the default
# distance between boxes of the stream)
SPACING = {"touching": BOX_SIZE + 0.001, "gap_3cm": BOX_SIZE + 0.03, "centre_11cm": 0.11, "centre_30cm": 0.30}
# neighbour offsets in units of the spacing, (along the belt, across the belt)
NEIGHBOURHOODS = {
    "free": [],
    "front": [(1, 0)],
    "back": [(-1, 0)],
    "front_back": [(1, 0), (-1, 0)],
    "left_right": [(0, 1), (0, -1)],
    "surrounded": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    "cluster_3x3": [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)],
}
COLORED = (BoxColor.RED, BoxColor.GREEN, BoxColor.BLUE)
TARGET_X = -0.6     # default position of the target along the belt [m]: upstream of Robot 1's picking area


@dataclass(frozen=True, eq=False)   # positions are numpy arrays: compare fields, not scenarios
class Scenario:
    name: str                                   # neighbourhood, see NEIGHBOURHOODS
    lateral: str                                # see LATERAL
    spacing: str                                # see SPACING
    target_color: BoxColor
    target: np.ndarray                          # (x, y) of the target box centre, world frame [m]
    neighbours: list[tuple[BoxColor, np.ndarray]] = field(default_factory=list)   # (color, (x, y))


def make_scenario(name: str, seed: int = 0, lateral: str = "centre", spacing: str = "touching",
                  x: float = TARGET_X, neighbour_color: BoxColor = BoxColor.GRAY) -> Scenario:
    """The layout `name` (a key of NEIGHBOURHOODS) with the target `lateral` (LATERAL) and the neighbours `spacing`
    (SPACING) away from it. The seed only picks the target's color: the same arguments give the same layout.
    ValueError if a box would not fit on the belt (e.g. side neighbours of a target at the edge)."""
    if name not in NEIGHBOURHOODS or lateral not in LATERAL or spacing not in SPACING:
        raise ValueError(f"unknown layout {name!r} / {lateral!r} / {spacing!r}")
    rng = np.random.default_rng(seed)
    target_color = COLORED[rng.integers(len(COLORED))]
    centre_y = 0.2                               # belt centre line (assets/scene.g)
    target = np.array([x, centre_y + LATERAL[lateral]])
    d = SPACING[spacing]
    neighbours = [(neighbour_color, target + [i * d, j * d]) for i, j in NEIGHBOURHOODS[name]]
    max_offset = 0.15 - BOX_SIZE / 2 + 1e-9      # belt half width minus half a box
    for _, position in [(target_color, target)] + neighbours:
        if abs(position[1] - centre_y) > max_offset:
            raise ValueError(f"layout {name!r} with the target at the {lateral} and {spacing} spacing does not fit "
                             f"on the belt")
    return Scenario(name, lateral, spacing, target_color, target, neighbours)


def spawn_scenario(world: World, scenario: Scenario) -> tuple[str, list[str]]:
    """Put the layout on the belt; returns the frame names of the target and of the neighbours."""
    centre_y = belt_geometry(world.C).y
    spawn = world.conveyor.spawn_box
    target = spawn(scenario.target_color, x=scenario.target[0], y=scenario.target[1] - centre_y)
    neighbours = [spawn(color, x=p[0], y=p[1] - centre_y) for color, p in scenario.neighbours]
    return target, neighbours


def scenarios(seed: int = 0) -> list[Scenario]:
    """Every combination of NEIGHBOURHOODS x LATERAL x SPACING that fits on the belt ("free" once per lateral)."""
    result = []
    for name in NEIGHBOURHOODS:
        for lateral in LATERAL:
            for spacing in (SPACING if name != "free" else ["touching"]):
                try:
                    result.append(make_scenario(name, seed, lateral, spacing))
                except ValueError:
                    pass
    return result
