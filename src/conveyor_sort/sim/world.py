"""Simulation world: scene + conveyor + clock, advanced in fixed time steps."""
from typing import Callable

from conveyor_sort.sim.conveyor import Conveyor, ConveyorConfig
from conveyor_sort.sim.scene import load_scene


class World:
    def __init__(self, conveyor_config: ConveyorConfig | None = None, dt: float = 0.05):
        self.C = load_scene()
        self.conveyor = Conveyor(self.C, conveyor_config)
        self.dt = dt
        self.time = 0.0   # simulation time [s]; use this for Detection / Track timestamps

    def step(self) -> None:
        self.conveyor.step(self.dt)
        self.time += self.dt

    def run(self, duration: float, on_step: Callable[["World"], None] | None = None) -> None:
        """Advance for `duration` seconds of simulation time; `on_step(world)` is called after every step."""
        for _ in range(round(duration / self.dt)):
            self.step()
            if on_step:
                on_step(self)
