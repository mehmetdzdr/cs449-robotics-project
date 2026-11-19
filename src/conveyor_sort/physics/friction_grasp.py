"""Friction grasp in the BotOp physics simulation.

The Franka gripper commands (`gripperClose`) move the fingers until they are about 5 mm from the object and
then hold that position: they never squeeze, so there is no normal force and nothing is held by friction.
`gripperCloseGrasp` lifts objects without closing the fingers at all (a simulator shortcut), so it is not used.

Here the finger joints are made active and driven by the arm's joint PD controller, like the other joints:
a finger reference inside the object makes the controller press the fingers against it, and the object is
held only by friction between fingers and object.
"""
from dataclasses import dataclass

import numpy as np
import robotic as ry

from conveyor_sort.physics.botop import start_simulation
from conveyor_sort.sim.scene import ROBOT_PREFIX

FINGER_OPEN = 0.04      # finger joint at full opening (8 cm between the fingers) [m]
FINGER_SQUEEZE = 0.015  # finger reference while holding a 6 cm box: the fingers touch it at 0.030 and settle near 0.0255
HOLDING_MIN = 0.022     # finger position above this after squeezing: something is between the fingers
                        # (measured: 0.0255 holding a 6 cm box, 0.018 closing on nothing; the PD settles short of 0.015)
FRICTION = 1.0          # PhysX friction coefficient (fingers, box, table)
SQUEEZE_SETTLE = 0.5    # wait after the squeeze command [s]


def activate_fingers(C: ry.Config) -> dict[int, int]:
    """Make the finger joints part of the controlled joints; returns robot_id -> index of its finger joint in q."""
    fingers = [f"{prefix}panda_finger_joint1" for prefix in ROBOT_PREFIX.values()]
    C.selectJoints([n for n in C.getJointNames() if n not in fingers] + fingers)
    # getJointNames also lists the mimic joints (finger_joint2 follows finger_joint1); q does not contain them
    names = [n for n in C.getJointNames() if not n.endswith("panda_finger_joint2")]
    assert len(names) == len(C.getJointState())
    return {robot_id: names.index(f"{prefix}panda_finger_joint1") for robot_id, prefix in ROBOT_PREFIX.items()}


def set_friction(friction: float = FRICTION) -> None:
    """PhysX default friction; must be set before the simulation is started."""
    ry.params_add({"physx/defaultFriction": friction})


@dataclass
class LiftTrial:
    grasp_offset: float    # box centre minus gripper centre in z after squeezing [m] (0 = grasped at the centre)
    finger_q: float        # finger joint after squeezing [m]
    holding: bool          # finger_q says something is between the fingers
    lift: float            # box height gain 5 s after the lift [m] (target 0.15)
    gripper_lift: float    # gripper height gain over the same time [m]; lift < gripper_lift means the box slipped
    tilt: float            # box tilt after squeezing [deg]
    attached: bool         # box became a child of another frame (must never happen)


def _move_gripper(C, bot, robot_id, finger_index, position, duration, finger, align_with=None):
    gripper = f"{ROBOT_PREFIX[robot_id]}gripper"
    komo = ry.KOMO(C, 1, 10, 2, True)
    komo.addControlObjective([], 2, 1e0)
    komo.addControlObjective([], 0, 1e-2)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([1.0], ry.FS.position, [gripper], ry.OT.eq, [1e1], position)
    komo.addObjective([0.6, 1.0], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], [0, 0, 1])
    if align_with:
        komo.addObjective([0.6, 1.0], ry.FS.scalarProductXY, [gripper, align_with], ry.OT.eq, [1e1])
    komo.addObjective([1.0], ry.FS.qItself, [], ry.OT.eq, [1e1], [], 1)
    ry.NLP_Solver(komo.nlp(), verbose=0).solve()
    path = komo.getPath()
    path[:, finger_index] = finger                       # finger reference along the whole motion
    bot.move(path, [duration])
    bot.wait(C, forKeyPressed=False)


def run_lift_trial(mass: float = 0.1, friction: float = FRICTION, yaw_deg: float = 0.0, robot_id: int = 1,
                   finger_target: float = FINGER_SQUEEZE, with_box: bool = True) -> LiftTrial:
    """Static 6 cm box on the table: approach from above, descend to its centre, squeeze, lift 15 cm, hold 5 s.

    Every trial starts a fresh simulation (about 10 s of simulated time, in real time).
    """
    set_friction(friction)
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandasTable.g"))
    finger = activate_fingers(C)[robot_id]
    x = -0.3 if robot_id == 1 else 0.3
    box = C.addFrame("box").setShape(ry.ST.ssBox, [0.06, 0.06, 0.06, 0.005]).setColor([1, 0, 0])
    box.setPosition([x, 0.2, 0.64] if with_box else [x, 0.2, -1.0]).setMass(mass).setContact(1)
    box.setQuaternion([np.cos(np.radians(yaw_deg) / 2), 0, 0, np.sin(np.radians(yaw_deg) / 2)])
    box.setAttributes({"friction": friction})
    bot = start_simulation(C)
    bot.sync(C, 0.5)                                     # box falls 1 cm and settles on the table
    above = np.array([x, 0.2, 0.75])
    align = "box" if with_box else None
    _move_gripper(C, bot, robot_id, finger, above, 2.0, FINGER_OPEN, align)
    # read the box pose only now: right after the start the synced pose can still be the spawn pose
    grasp_at = C.getFrame("box").getPosition().copy() if with_box else np.array([x, 0.2, 0.63])
    _move_gripper(C, bot, robot_id, finger, grasp_at, 1.5, FINGER_OPEN, align)
    q = C.getJointState()
    q[finger] = finger_target
    bot.moveTo(q, 1.0)                                   # squeeze: fingers driven by the PD controller
    bot.wait(C, forKeyPressed=False)
    bot.sync(C, SQUEEZE_SETTLE)
    finger_q = float(C.getJointState()[finger])
    grasp_offset = float(C.getFrame("box").getPosition()[2] - C.getFrame(f"{ROBOT_PREFIX[robot_id]}gripper").getPosition()[2])
    tilt = float(np.degrees(np.arccos(np.clip(C.getFrame("box").getRotationMatrix()[2, 2], -1, 1))))
    gripper = C.getFrame(f"{ROBOT_PREFIX[robot_id]}gripper")
    gripper_start = gripper.getPosition()[2]
    box_start = C.getFrame("box").getPosition()[2]
    _move_gripper(C, bot, robot_id, finger, grasp_at + [0, 0, 0.15], 2.0, finger_target)
    bot.sync(C, 5.0)
    lift = float(C.getFrame("box").getPosition()[2] - box_start) if with_box else 0.0
    gripper_lift = float(gripper.getPosition()[2] - gripper_start)
    attached = C.getFrame("box").getParent() is not None
    del bot
    return LiftTrial(grasp_offset, finger_q, finger_q > HOLDING_MIN, lift, gripper_lift, tilt, attached)
