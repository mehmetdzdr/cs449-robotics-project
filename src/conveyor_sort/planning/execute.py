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


def grasp(world: World, robot_id: int, box: str | None = None, tolerance: float = 0.01) -> str | None:
    """Kinematic grasp: attach a box to the gripper if the gripper centre is within `tolerance` of the box centre.

    Without `box`, the box on the belt closest to the gripper is taken. Returns the grasped box name, or None.
    """
    gripper = world.C.getFrame(f"{ROBOT_PREFIX[robot_id]}gripper")

    def distance(name: str) -> float:
        return float(np.linalg.norm(gripper.getPosition() - world.C.getFrame(name).getPosition()))

    if box is None:
        box = min(world.conveyor.on_belt(), key=distance, default=None)
    if box is None or distance(box) > tolerance:
        return None
    world.conveyor.attach(box, gripper.name)
    return box


def release(world: World, box: str, region: str) -> None:
    """Kinematic release: the box stays where it is and becomes a child of the region."""
    world.conveyor.attach(box, region)
