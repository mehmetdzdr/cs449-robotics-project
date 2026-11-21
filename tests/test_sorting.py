"""The sorting loop itself runs on physics: test_loops_physics.py, test_shared_area_physics.py (#113)."""
import pytest

from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots


def test_the_loop_needs_the_physics_backend():
    with pytest.raises(ValueError, match="physics"):
        run_robots(World(ConveyorConfig(max_boxes=0)), lambda: [], 1.0)


def test_clearing_steps_call_the_start_function():
    """A loop variable named `start` in run_clearing made `start` local there: every physics clearing step failed
    with UnboundLocalError (the physics runs are slow tests; this checks the code itself)."""
    import conveyor_sort.sorting as sorting
    nested = {c.co_name: c for c in sorting.run_robots.__code__.co_consts if hasattr(c, "co_name")}
    assert "start" in nested and "start" in nested["run_clearing"].co_freevars
    assert "start" not in nested["run_clearing"].co_varnames
