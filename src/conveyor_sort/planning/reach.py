"""Reachability check by inverse kinematics (KOMO)."""
import numpy as np
import robotic as ry

from conveyor_sort.sim.scene import ROBOT_PREFIX


def is_reachable(C: ry.Config, robot_id: int, position: np.ndarray) -> bool:
    """True if the robot's gripper can reach `position` (world frame) with a top-down grasp."""
    gripper = f"{ROBOT_PREFIX[robot_id]}gripper"
    target = C.addFrame("_reach_target").setPosition(position)
    try:
        komo = ry.KOMO(C, 1, 1, 0, False)
        komo.addControlObjective([], 0, 1e-1)
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
        komo.addObjective([], ry.FS.positionDiff, [gripper, target.name], ry.OT.eq, [1e1])
        komo.addObjective([], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], [0, 0, 1])
        ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
        return bool(ret.feasible)
    finally:
        C.delFrame(target)


LIFT_HEIGHT = 0.15     # a grasped box is lifted this far [m]
EDGE_MARGIN = 0.1      # distance to keep from the limit of the workspace along the belt [m]


def can_pick(C: ry.Config, robot_id: int, position: np.ndarray) -> bool:
    """True if the robot can grasp a box at `position` and lift it.

    At the edge of the workspace a box can be reached but not lifted, and the arm is then left holding it.
    So the points at lift height `EDGE_MARGIN` upstream and downstream must be reachable too.
    """
    return (is_reachable(C, robot_id, position)
            and is_reachable(C, robot_id, position + [EDGE_MARGIN, 0, LIFT_HEIGHT])
            and is_reachable(C, robot_id, position + [-EDGE_MARGIN, 0, LIFT_HEIGHT]))
