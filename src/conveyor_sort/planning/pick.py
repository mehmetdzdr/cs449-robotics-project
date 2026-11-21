"""KOMO trajectories for pick and place: reach a box, lift it, place it in a region, return home."""
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import robotic as ry

from conveyor_sort.interfaces import Track
from conveyor_sort.sim.conveyor import BOX_SIZE
from conveyor_sort.sim.scene import HOME_Q, ROBOT_PREFIX, arm_joints


@dataclass
class Plan:
    path: np.ndarray        # (steps, 14 or 16) joint waypoints for both arms, from the current state to the goal
    feasible: bool          # all constraints satisfied (reach, joint limits, no collisions)
    planning_time: float    # [s]


ARM_JOINT_NAMES = [f"{prefix}panda_joint{i}" for prefix in ROBOT_PREFIX.values() for i in range(1, 8)]
_FULL_STATE: dict[int, tuple[np.ndarray, list[int]]] = {}   # id(komo) -> (full q, planned indices in it), see _komo
# Within own_arm_only, the hand and forearm of the planning arm keep this distance to the other arm's hand and forearm
# while that one is over or near the belt (gripper within NEAR_BELT of its centre line) (#154). The collision term
# alone only keeps distance >= 0 at the waypoints: a home path planned next to Robot 2's arm (squeezing a box on Robot
# 1's half of the belt) touched it when executed. With 3 cm: no contact in the same run, closest 2.8 cm. It costs
# planning time (pick 0.04 s -> 0.12 s), which passes in real time on physics: not added while the other arm is at
# home or over its own regions, far from the belt.
MARGIN = 0.03      # [m]
NEAR_BELT = 0.25   # [m]
# A place that starts close to the other arm resting at home (grippers within CLOSE, xy; the other one within NEAR_HOME
# of the belt centre line) keeps MARGIN_HOME from it (#159): after a grasp at the far edge of the belt near its middle,
# Robot 1's place motion passed Robot 2's resting wrist at distance 0 and touched it (2 of 30 runs in #108). Only the
# place: with the margin also on the pick, the same far-edge grasps could not be planned (#106).
MARGIN_HOME, NEAR_HOME, CLOSE = 0.022, 0.3, 0.35   # [m]
_OWN_ARM_ONLY: set[int] = set()                               # id(C) of configs planned with own_arm_only


@contextmanager
def own_arm_only(C: ry.Config):
    """Within this block, the planners optimize only the planning robot's arm joints; the other arm is fixed at
    its current pose. Otherwise KOMO plans both arms and may move the other one out of the way (seen: by 0.17
    rad, the executed path then went 14 cm into the other arm), while only the planning robot's joints are
    executed (#141). Used on the physics backend while the other arm stands still; while it moves, the plan
    is checked against its real motion instead (`physics.arms.ArmReferences.clearance`)."""
    _OWN_ARM_ONLY.add(id(C))
    try:
        yield
    finally:
        _OWN_ARM_ONLY.discard(id(C))


def _komo(C: ry.Config, steps: int, robot_id: int, resting_margin: bool = False) -> ry.KOMO:
    """KOMO over the arm joints (both arms, or only `robot_id`'s within `own_arm_only`). With active finger
    joints (physics backend, q has 16 entries) the fingers are not decision variables but keep their current
    value (they are not the planner's to move, see physics.arms); `_solve` puts the joints that are not planned
    back into the path. An equality constraint on the fingers instead let the solver trade finger motion against
    the other terms and made some lifts infeasible (in about 1 of 10 trials)."""
    q = C.getJointState()
    own_only = id(C) in _OWN_ARM_ONLY
    if len(q) <= 14 and not own_only:
        return ry.KOMO(C, 1, steps, 2, True)
    planned = arm_joints(C, robot_id) if own_only else arm_joints(C, 1) + arm_joints(C, 2)
    C.selectJoints([f"{ROBOT_PREFIX[robot_id]}panda_joint{i}" for i in range(1, 8)] if own_only else ARM_JOINT_NAMES)
    try:
        komo = ry.KOMO(C, 1, steps, 2, True)
    finally:
        C.selectJoints(ARM_JOINT_NAMES)       # the default selection (both arms)
        if len(q) > 14:
            from conveyor_sort.physics.friction_grasp import activate_fingers
            activate_fingers(C)                 # ... plus the fingers
        C.setJointState(q)
    _FULL_STATE[id(komo)] = q, planned
    other = next(r for r in ROBOT_PREFIX if r != robot_id)
    belt_y = C.getFrame("belt").getPosition()[1]
    other_gripper = C.getFrame(f"{ROBOT_PREFIX[other]}gripper").getPosition()
    off_centre = abs(other_gripper[1] - belt_y)
    close = np.linalg.norm(C.getFrame(f"{ROBOT_PREFIX[robot_id]}gripper").getPosition()[:2] - other_gripper[:2]) < CLOSE
    margin = (MARGIN if off_centre < NEAR_BELT else
              MARGIN_HOME if resting_margin and close and off_centre < NEAR_HOME else 0.0)
    if own_only and margin:
        for own_shape in _arm_shapes(C, robot_id):
            for other_shape in _arm_shapes(C, other):
                komo.addObjective([], ry.FS.negDistance, [own_shape, other_shape], ry.OT.ineq, [1e1], [-margin])
    return komo


def _arm_shapes(C: ry.Config, robot_id: int) -> list[str]:
    """Collision shapes of a robot's forearm, hand and fingers (where two arms reaching over the belt meet)."""
    prefix, names = ROBOT_PREFIX[robot_id], set(C.getFrameNames())
    return [n for n in [f"{prefix}panda_coll{i}" for i in range(5, 8)] + [f"{prefix}palm", f"{prefix}finger1", f"{prefix}finger2"]
            if n in names]


def plan_pick(C: ry.Config, robot_id: int, box: str, steps: int = 20) -> Plan:
    """Plan from the current joint state to a grasp pose at the (stationary) box.

    At the end the gripper centre is at the box centre, pointing straight down, fingers on two
    opposite faces. During the last 20 % of the motion the gripper stays above the box and descends
    vertically, so the fingers do not sweep through the box or its neighbours.
    """
    gripper = f"{ROBOT_PREFIX[robot_id]}gripper"
    komo = _komo(C, steps, robot_id)
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
    komo = _komo(C, steps, robot_id)
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
    path = komo.getPath()
    if id(komo) in _FULL_STATE:            # not all joints planned (see _komo): re-insert the fixed ones
        full_q, arms = _FULL_STATE.pop(id(komo))
        full = np.tile(full_q, (len(path), 1))
        full[:, arms] = path
        path = full
    return Plan(path, bool(result.feasible), time.perf_counter() - start)


def plan_place(C: ry.Config, robot_id: int, box: str, region: str, steps: int = 20, clearance: float = 0.0,
               offset: tuple[float, float] = (0.0, 0.0)) -> Plan:
    """Plan carrying the grasped box to the centre of `region` (or `offset` from it, e.g. a slot) and setting it
    down (vertical descent at the end), `clearance` above its resting height."""
    gripper = f"{ROBOT_PREFIX[robot_id]}gripper"
    resting = [*offset, C.getFrame(region).getSize()[2] / 2 + BOX_SIZE / 2 + 0.001 + clearance]   # box centre, region frame
    komo = _komo(C, steps, robot_id, resting_margin=True)
    komo.addControlObjective([], 2, 1e0)
    komo.addControlObjective([], 0, 1e-2)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], [0, 0, 1])                # keep the box upright
    komo.addObjective([0.8, 1.0], ry.FS.positionRel, [box, region], ry.OT.eq, [[1e1, 0, 0], [0, 1e1, 0]], resting)
    komo.addObjective([1.0], ry.FS.positionRel, [box, region], ry.OT.eq, [1e1], resting)
    komo.addObjective([1.0], ry.FS.qItself, [], ry.OT.eq, [1e1], [], 1)
    return _solve(komo)


def plan_home(C: ry.Config, robot_id: int, steps: int = 20) -> Plan:
    """Plan the robot's return to its retracted home pose; the other arm keeps its current pose."""
    q = C.getJointState()
    komo = _komo(C, steps, robot_id)
    komo.addControlObjective([], 2, 1e0)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    for r, prefix in ROBOT_PREFIX.items():      # by joint name: with active fingers q has 16 entries (KOMO's 18)
        if r == robot_id or id(C) not in _OWN_ARM_ONLY:
            target = HOME_Q if r == robot_id else q[arm_joints(C, r)]
            komo.addObjective([1.0], ry.FS.qItself, [f"{prefix}panda_joint{i}" for i in range(1, 8)], ry.OT.eq, [1e1], target)
    komo.addObjective([1.0], ry.FS.qItself, [], ry.OT.eq, [1e1], [], 1)
    return _solve(komo)


def plan_moving_pick(C: ry.Config, robot_id: int, track: Track, now: float, pick_time: float, steps: int = 20,
                     across_belt: bool = False) -> Plan:
    """Plan a grasp of a moving box: the gripper meets the box at `pick_time` where the track predicts it.

    The path must be executed in exactly `pick_time - now` seconds. During the last 20 % the gripper
    is already above the box and follows its predicted motion while descending, so the fingers come
    down around the moving box instead of sweeping through it. Infeasible if the predicted positions
    are out of reach.

    The fingers close along the belt direction (on the box faces at -x and +x), so a neighbour closer than
    about 10 cm in front of or behind the box is hit. With `across_belt` they close on the faces at -y and +y.
    """
    gripper = f"{ROBOT_PREFIX[robot_id]}gripper"
    target = C.addFrame("_pick_target").setPosition(track.predict(pick_time))   # world-aligned, like the boxes
    try:
        komo = _komo(C, steps, robot_id)
        komo.addControlObjective([], 2, 1e0)
        komo.addControlObjective([], 0, 1e-2)
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
        komo.addObjective([0.8, 1.0], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], [0, 0, 1])
        komo.addObjective([0.8, 1.0], ry.FS.scalarProductXX if across_belt else ry.FS.scalarProductXY,
                          [gripper, target.name], ry.OT.eq, [1e1])
        for k in range(round(0.8 * steps), steps + 1):          # follow the box in xy, waypoint by waypoint
            phase = k / steps
            predicted = track.predict(now + phase * (pick_time - now))
            komo.addObjective([phase], ry.FS.position, [gripper], ry.OT.eq, [[1e1, 0, 0], [0, 1e1, 0]], predicted)
        komo.addObjective([1.0], ry.FS.positionDiff, [gripper, target.name], ry.OT.eq, [1e1])
        return _solve(komo)
    finally:
        C.delFrame(target)
