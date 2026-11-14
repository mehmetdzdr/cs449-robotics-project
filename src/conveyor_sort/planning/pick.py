"""KOMO trajectory for reaching a box with a top-down grasp pose."""
import time
from dataclasses import dataclass

import numpy as np
import robotic as ry

from conveyor_sort.sim.scene import ROBOT_PREFIX


@dataclass
class Plan:
    path: np.ndarray        # (steps, 14) joint waypoints for both arms, from the current state to the goal
    feasible: bool          # all constraints satisfied (reach, joint limits, no collisions)
    planning_time: float    # [s]


def plan_pick(C: ry.Config, robot_id: int, box: str, steps: int = 20) -> Plan:
    """Plan from the current joint state to a grasp pose at the (stationary) box.

    At the end the gripper centre is at the box centre, pointing straight down, fingers on two
    opposite faces. During the last 20 % of the motion the gripper stays above the box and descends
    vertically, so the fingers do not sweep through the box or its neighbours.
    """
    gripper = f"{ROBOT_PREFIX[robot_id]}gripper"
    komo = ry.KOMO(C, 1, steps, 2, True)
    komo.addControlObjective([], 2, 1e0)                                        # smooth: small accelerations
    komo.addControlObjective([], 0, 1e-2)                                       # stay near the start pose
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([1.0], ry.FS.positionDiff, [gripper, box], ry.OT.eq, [1e1])
    komo.addObjective([0.8, 1.0], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], [0, 0, 1])
    komo.addObjective([0.8, 1.0], ry.FS.scalarProductXY, [gripper, box], ry.OT.eq, [1e1])
    komo.addObjective([0.8, 1.0], ry.FS.positionRel, [gripper, box], ry.OT.eq, [[1e1, 0, 0], [0, 1e1, 0]])
    komo.addObjective([1.0], ry.FS.qItself, [], ry.OT.eq, [1e1], [], 1)         # stop at the goal

    start = time.perf_counter()
    result = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
    return Plan(komo.getPath(), bool(result.feasible), time.perf_counter() - start)
