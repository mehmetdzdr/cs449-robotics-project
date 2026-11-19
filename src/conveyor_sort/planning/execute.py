"""Execute planned joint paths in the simulation world; kinematic grasp (box is attached to the gripper)."""
from typing import Callable, Iterator

import numpy as np

from conveyor_sort.sim.scene import ROBOT_PREFIX
from conveyor_sort.sim.world import World


def follow_path(world: World, path: np.ndarray, duration: float, robot_id: int | None = None) -> Iterator[None]:
    """Generator form of `execute_path`: sets the joints for one simulation step, then yields for the caller to
    step the world. With `robot_id` only that robot's joints are moved, so both arms can follow their own paths.

    Physics backend: the path goes to BotOp's controller as a joint reference (`physics.arms`) that ends at
    `world.time + duration` on the simulation clock (planning time that passed meanwhile is taken from the path,
    so a pick still arrives at its pick time); the generator yields until that time.
    """
    if world.bot is not None:
        end = world.time + duration
        world.arms.set_path(robot_id, np.asarray(path), end + world._t0)
        while world.time < end - 1e-6:
            yield
        return
    joints = slice((robot_id - 1) * 7, robot_id * 7) if robot_id else slice(None)
    waypoints = np.vstack([world.C.getJointState(), path])[:, joints]   # start from the current joint state
    n_steps = max(1, round(duration / world.dt))
    for k in range(1, n_steps + 1):
        s = k / n_steps * (len(waypoints) - 1)               # linear interpolation between waypoints
        i = min(int(s), len(waypoints) - 2)
        q = world.C.getJointState()
        q[joints] = waypoints[i] + (s - i) * (waypoints[i + 1] - waypoints[i])
        world.C.setJointState(q)
        yield


def execute_path(world: World, path: np.ndarray, duration: float,
                 on_step: Callable[[World], None] | None = None) -> None:
    """Move the arms along `path` in `duration` seconds while the world (conveyor, clock) keeps running."""
    for _ in follow_path(world, path, duration):
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
