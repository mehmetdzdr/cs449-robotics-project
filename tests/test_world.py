import pytest

from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World


def test_run_advances_clock_and_calls_back_every_step():
    world = World(dt=0.1)
    times = []
    world.run(2.0, lambda w: times.append(w.time))
    assert world.time == pytest.approx(2.0)
    assert len(times) == 20


def test_long_run_reaches_steady_state():
    # belt 2.2 m, speed 0.2 m/s, one box every 0.4 m -> 5 boxes on the belt, the rest fell off the end
    world = World(ConveyorConfig(speed=0.2, spacing=0.4), dt=0.05)
    world.run(30.0)
    assert len(world.conveyor.on_belt()) in (5, 6)
    assert world.conveyor.missed in (9, 10)
