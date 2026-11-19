import time

import numpy as np
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


# ---------------------------------------------------------------- physics backend (BotOp)

@pytest.fixture
def physics_world():
    """Factory for physics worlds; every world is closed after the test, also when it fails."""
    worlds = []

    def make(*args, **kwargs) -> World:
        worlds.append(World(*args, physics=True, **kwargs))
        return worlds[-1]

    yield make
    for world in worlds:
        world.close()


def test_physics_clock_comes_from_botop(physics_world):
    world = physics_world(ConveyorConfig(speed=0.1, spacing=100.0))
    assert world.time == 0.0
    times = []
    world.run(2.0, lambda w: times.append(w.time))
    assert world.time == pytest.approx(2.0, abs=world.dt)                     # +- one step
    assert world.bot.get_t() - world._t0 >= world.time                         # BotOp's clock keeps running
    steps = np.diff([0.0] + times)
    assert steps.min() >= world.dt - 1e-6 and steps.mean() < 1.5 * world.dt  # a step is dt or a little more
    box = world.conveyor.on_belt()[0]                                          # the belt moved by the elapsed time
    x_start = world.conveyor.geometry.x_start + 0.03
    assert world.C.getFrame(box).getPosition()[0] - x_start == pytest.approx(0.1 * world.time, abs=1e-6)
    world.close()
    assert world.bot is None


@pytest.mark.parametrize("hyper_speed", [1, 5, 10])
def test_hyper_speed_real_time_factor(physics_world, hyper_speed):
    # measured: 1.00 / 5.0 to 5.3 / 9.7 to 10.2; steps get longer than dt above 1 (one sync costs about 40 ms
    # wall time): mean step 0.05 / 0.20 / 0.38 s
    world = physics_world(hyper_speed=hyper_speed)
    start = time.time()
    world.run(5.0)
    factor = world.time / (time.time() - start)
    print(f"hyperSpeed {hyper_speed}: real-time factor {factor:.2f}")
    assert 0.7 * hyper_speed < factor < 1.5 * hyper_speed


def test_same_seed_gives_same_boxes_at_the_same_time(physics_world):
    # the free-running clock ends each run at a slightly different time (raw spread of x after run(10):
    # 1 mm at hyperSpeed 1, 13 mm at 5, 31 mm at 10); at the same time the boxes are at the same place
    runs = []
    for _ in range(5):
        world = physics_world(ConveyorConfig(speed=0.1, spacing=0.3, seed=0), hyper_speed=10)
        world.run(10.0)
        assert world.time == pytest.approx(10.0, abs=0.5)
        runs.append({n: world.C.getFrame(n).getPosition()[0] - 0.1 * (world.time - 10.0) for n in world.conveyor.on_belt()})
        world.close()
    assert all(set(r) == set(runs[0]) for r in runs) and len(runs[0]) == 4
    spread = max(np.ptp([r[n] for r in runs]) for n in runs[0])
    print(f"same seed x5: max x spread at the same time {spread * 1000:.2f} mm")
    assert spread < 1e-3
