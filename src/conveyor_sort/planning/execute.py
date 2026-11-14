"""Execute planned joint paths in the simulation world; kinematic grasp (box is attached to the gripper)."""
from typing import Callable

import numpy as np

from conveyor_sort.sim.scene import ROBOT_PREFIX
from conveyor_sort.sim.world import World


def execute_path(world: World, path: np.ndarray, duration: float,
                 on_step: Callable[[World], None] | None = None) -> None:
    """Move the arms along `path` in `duration` seconds while the world (conveyor, clock) keeps running."""
    waypoints = np.vstack([world.C.getJointState(), path])   # start from the current joint state
    n_steps = max(1, round(duration / world.dt))
    for k in range(1, n_steps + 1):
        s = k / n_steps * (len(waypoints) - 1)               # linear interpolation between waypoints
        i = min(int(s), len(waypoints) - 2)
        world.C.setJointState(waypoints[i] + (s - i) * (waypoints[i + 1] - waypoints[i]))
        world.step()
        if on_step:
            on_step(world)


def grasp(world: World, robot_id: int, box: str, tolerance: float = 0.01) -> bool:
    """Kinematic grasp: attach the box to the gripper if the gripper centre is within `tolerance` of the box centre."""
    gripper = f"{ROBOT_PREFIX[robot_id]}gripper"
    distance = np.linalg.norm(world.C.getFrame(gripper).getPosition() - world.C.getFrame(box).getPosition())
    if distance > tolerance:
        return False
    world.conveyor.attach(box, gripper)
    return True
