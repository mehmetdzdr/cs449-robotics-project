"""Simulation world: scene + conveyor + clock, advanced in time steps.

Two backends:
- kinematic (default): a deterministic clock, every step is exactly `dt`.
- physics (`physics=True`): BotOp's physics simulation runs in its own thread and its clock does not stop
  (there is no stepping mode). A step waits until `dt` of simulation time has passed (`bot.sync`, which waits
  in wall-clock time) and reads the clock from `bot.get_t()`, so steps are `dt` or a little longer and two
  runs with the same seed are not identical. The conveyor is advanced by the simulation time that actually
  passed; boxes are physics bodies driven by the belt (see conveyor.py). Paths and grasps are still kinematic
  until #90 and #91. `close()` a physics World before
  starting the next one.
"""
from typing import Callable

import robotic as ry

from conveyor_sort.sim.conveyor import Conveyor, ConveyorConfig
from conveyor_sort.sim.scene import load_scene


class World:
    def __init__(self, conveyor_config: ConveyorConfig | None = None, dt: float = 0.05,
                 physics: bool = False, hyper_speed: float = 1.0, friction: float | None = None):
        self.C = load_scene()
        self.conveyor = Conveyor(self.C, conveyor_config, physics=physics)
        self.dt = dt
        self.time = 0.0   # simulation time [s]; use this for Detection / Track timestamps
        self.bot: ry.BotOp | None = None
        self.arms = None                  # physics backend: physics.arms.ArmReferences, used by follow_path
        self.fingers: dict[int, int] = {}  # physics backend: robot_id -> index of its finger joint in q
        self.hyper_speed = hyper_speed
        if physics:
            from conveyor_sort.physics.arms import ArmReferences
            from conveyor_sort.physics.botop import start_simulation
            from conveyor_sort.physics.friction_grasp import FINGER_OPEN, FRICTION, activate_fingers, set_friction
            set_friction(FRICTION if friction is None else friction)
            self.fingers = activate_fingers(self.C)
            self.bot = start_simulation(self.C, hyper_speed)
            self.conveyor.bot = self.bot
            self.bot.sync(self.C, 0.0)
            self._t0 = self.bot.get_t()   # BotOp's clock already runs during start-up; World.time starts at 0
            self.arms = ArmReferences(self.bot, self.C, self.fingers, FINGER_OPEN)

    def step(self) -> None:
        if self.bot is None:
            self.conveyor.step(self.dt)
            self.time += self.dt
            return
        end = self.time + self.dt
        while self.time < end - 1e-6:
            self.bot.sync(self.C, (end - self.time) / self.hyper_speed)
            time = self.bot.get_t() - self._t0
            self.conveyor.step(time - self.time)
            self.time = time

    def run(self, duration: float, on_step: Callable[["World"], None] | None = None) -> None:
        """Advance for `duration` seconds of simulation time; `on_step(world)` is called after every step."""
        if self.bot is None:
            for _ in range(round(duration / self.dt)):
                self.step()
                if on_step:
                    on_step(self)
            return
        end = self.time + duration
        while self.time < end - self.dt / 2:
            self.step()
            if on_step:
                on_step(self)

    def close(self) -> None:
        """Stop the physics simulation (no-op for the kinematic backend)."""
        if self.bot is not None:
            self.conveyor.bot = None      # the last reference must go, or BotOp lives on (and crashes at exit)
            self.arms = None
            del self.bot
            self.bot = None
