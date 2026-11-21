"""Load the simulation scene: RAI's two-Panda table plus the project additions in assets/scene.g."""
from pathlib import Path

import numpy as np
import robotic as ry

ASSETS = Path(__file__).resolve().parents[3] / "assets"

# robot_id (see interfaces.Assignment) -> frame-name prefix in the RAI config
ROBOT_PREFIX = {1: "l_", 2: "r_"}
TOP_CAMERA = "cameraTop"

# Base poses (#154): both robots at the middle of the belt (x = 0), on opposite sides, 0.5 m from its centre line
# (y = 0.2), facing it. Robot 2 stands on the table extension of assets/scene.g. Quaternions are (w, x, y, z).
ROBOT_BASES = {1: ([0.0, -0.3, 0.6], [0.7071068, 0, 0, 0.7071068]),
               2: ([0.0, 0.7, 0.6], [0.7071068, 0, 0, -0.7071068])}

# Retracted arm pose (7 joints): keeps both arms out of the top camera's view of the belt area.
HOME_Q = np.array([0.0, -1.2, 0.0, -2.7, 0.0, 1.6, -0.5])


def arm_joints(C: ry.Config, robot_id: int) -> list[int]:
    """Indices of a robot's 7 arm joints in q: 0..6 and 7..13, but 0..6 and 8..14 when the finger joints are
    active (each finger comes right after its own arm in q, see physics.friction_grasp.activate_fingers)."""
    names = [n for n in C.getJointNames() if not n.endswith("panda_finger_joint2")]   # mimic joints are not in q
    if len(names) != len(C.getJointState()):
        names = [n for n in names if "finger" not in n]
    return [i for i, n in enumerate(names) if n.startswith(ROBOT_PREFIX[robot_id]) and "finger" not in n]


def load_scene() -> ry.Config:
    # no shadows in rendered images (#154): RAI's light falls from the far side of the belt, so Robot 2's arm, now
    # there, shadowed the downstream belt, and gray boxes in its shadow were as dark as the belt (HSV missed 5 of 125
    # visible boxes in the random scenes of test_hsv; without shadows none)
    ry.params_add({"Render/renderShadow": False})
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandasTable.g"))
    C.addFile(str(ASSETS / "scene.g"))
    # like RAI's table, the extension hangs below `origin` (a .g file cannot name frames of another file as parents):
    # the robot base capsules (`contact: -2`) then do not collide with it
    C.getFrame("table_far").setParent(C.getFrame("origin"), True)
    for robot_id, (position, quaternion) in ROBOT_BASES.items():   # RAI's pandasTable.g puts them side by side
        C.getFrame(f"{ROBOT_PREFIX[robot_id]}panda_base").setPosition(position).setQuaternion(quaternion)
    C.setJointState(np.concatenate([HOME_Q, HOME_Q]))
    return C
