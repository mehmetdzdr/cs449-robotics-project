"""KOMO trajectories for pick and place: reach a box, lift it, place it in a region, return home."""
import time
from dataclasses import dataclass

import numpy as np
import robotic as ry

from conveyor_sort.sim.conveyor import BOX_SIZE
from conveyor_sort.sim.scene import HOME_Q, ROBOT_PREFIX


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
    return _solve(komo)


def plan_lift(C: ry.Config, robot_id: int, height: float = 0.15, steps: int = 10) -> Plan:
    """Plan a straight vertical lift of the gripper (and a box attached to it) by `height`."""
    gripper = f"{ROBOT_PREFIX[robot_id]}gripper"
    target = C.getFrame(gripper).getPosition() + [0, 0, height]
    komo = ry.KOMO(C, 1, steps, 2, True)
    komo.addControlObjective([], 2, 1e0)
    komo.addControlObjective([], 0, 1e-2)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], [0, 0, 1])
    komo.addObjective([], ry.FS.position, [gripper], ry.OT.eq, [[1e1, 0, 0], [0, 1e1, 0]], target)   # stay on the vertical line
    komo.addObjective([1.0], ry.FS.position, [gripper], ry.OT.eq, [1e1], target)
    komo.addObjective([1.0], ry.FS.qItself, [], ry.OT.eq, [1e1], [], 1)
    return _solve(komo)


def _solve(komo: ry.KOMO) -> Plan:
    start = time.perf_counter()
    result = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
    return Plan(komo.getPath(), bool(result.feasible), time.perf_counter() - start)


def plan_place(C: ry.Config, robot_id: int, box: str, region: str, steps: int = 20) -> Plan:
    """Plan carrying the grasped box to the centre of `region` and setting it down (vertical descent at the end)."""
    gripper = f"{ROBOT_PREFIX[robot_id]}gripper"
    resting = [0, 0, C.getFrame(region).getSize()[2] / 2 + BOX_SIZE / 2 + 0.001]   # box centre in the region frame
    komo = ry.KOMO(C, 1, steps, 2, True)
    komo.addControlObjective([], 2, 1e0)
    komo.addControlObjective([], 0, 1e-2)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], [0, 0, 1])                # keep the box upright
    komo.addObjective([0.8, 1.0], ry.FS.positionRel, [box, region], ry.OT.eq, [[1e1, 0, 0], [0, 1e1, 0]])
    komo.addObjective([1.0], ry.FS.positionRel, [box, region], ry.OT.eq, [1e1], resting)
    komo.addObjective([1.0], ry.FS.qItself, [], ry.OT.eq, [1e1], [], 1)
    return _solve(komo)


def plan_home(C: ry.Config, robot_id: int, steps: int = 20) -> Plan:
    """Plan the robot's return to its retracted home pose; the other arm keeps its current pose."""
    target = C.getJointState()
    target[(robot_id - 1) * 7: robot_id * 7] = HOME_Q
    komo = ry.KOMO(C, 1, steps, 2, True)
    komo.addControlObjective([], 2, 1e0)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([1.0], ry.FS.qItself, [], ry.OT.eq, [1e1], target)
    komo.addObjective([1.0], ry.FS.qItself, [], ry.OT.eq, [1e1], [], 1)
    return _solve(komo)
