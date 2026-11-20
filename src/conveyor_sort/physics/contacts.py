"""Arm-arm proximity on the physics backend (#95).

BotOp exposes no contact list of its PhysX simulation, so contacts are read from the poses the simulation
reports: after every sync, C holds the simulated joint states and box poses, and RAI's collision check on C
gives the distance between the two arms' collision shapes (negative: penetration, i.e. contact).
"""
import robotic as ry

from conveyor_sort.sim.scene import ROBOT_PREFIX

MARGIN = 0.2    # pairs farther apart than this are not reported [m]


def arm_proximity(C: ry.Config, margin: float = MARGIN) -> tuple[float, list[tuple[str, str, float]]]:
    """(smallest distance between a frame of Robot 1 and one of Robot 2, the pairs in contact); inf if no pair
    is closer than `margin`. A box a gripper holds counts as the box, not as part of the arm."""
    left, right = ROBOT_PREFIX[1], ROBOT_PREFIX[2]
    C.computeCollisions()
    pairs = [(a, b, d) for a, b, d in C.getCollisions(margin)
             if (a.startswith(left) and b.startswith(right)) or (a.startswith(right) and b.startswith(left))]
    return min((d for _, _, d in pairs), default=float("inf")), [p for p in pairs if p[2] <= 0.0]
