"""Execute planned joint paths on the BotOp physics backend (the kinematic path playback was removed in #114)."""
from typing import Callable, Iterator

import numpy as np

from conveyor_sort.sim.world import World


def follow_path(world: World, path: np.ndarray, duration: float, robot_id: int | None = None) -> Iterator[None]:
    """Generator form of `execute_path`: the path goes to BotOp's controller as a joint reference (`physics.arms`)
    that ends at `world.time + duration` on the simulation clock (planning time that passed meanwhile is taken from
    the path, so a pick still arrives at its pick time); yields until that time, the caller steps the world. With
    `robot_id` only that robot's joints follow the path, so both arms can follow their own paths."""
    if world.bot is None:
        raise ValueError("follow_path needs the physics backend (World(..., physics=True)); the kinematic path "
                         "playback was removed (#114)")
    end = world.time + duration
    world.arms.set_path(robot_id, np.asarray(path), end + world._t0)
    while world.time < end - 1e-6:
        yield


def execute_path(world: World, path: np.ndarray, duration: float,
                 on_step: Callable[[World], None] | None = None) -> None:
    """Move the arms along `path` in `duration` seconds while the world (conveyor, clock) keeps running (physics)."""
    for _ in follow_path(world, path, duration):
        world.step()
        if on_step:
            on_step(world)

