"""Load the simulation scene: RAI's two-Panda table plus the project additions in assets/scene.g."""
from pathlib import Path

import numpy as np
import robotic as ry

ASSETS = Path(__file__).resolve().parents[3] / "assets"

# robot_id (see interfaces.Assignment) -> frame-name prefix in the RAI config
ROBOT_PREFIX = {1: "l_", 2: "r_"}
TOP_CAMERA = "cameraTop"

# Retracted arm pose (7 joints): keeps both arms out of the top camera's view of the belt area.
HOME_Q = np.array([0.0, -1.2, 0.0, -2.7, 0.0, 1.6, -0.5])


def load_scene() -> ry.Config:
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandasTable.g"))
    C.addFile(str(ASSETS / "scene.g"))
    C.setJointState(np.concatenate([HOME_Q, HOME_Q]))
    return C
