"""Command mode without a pick (#113: picks run on physics, see test_loops_physics.py)."""
import pytest

from conveyor_sort.command.mode import CommandMode
from conveyor_sort.interfaces import BoxColor
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.world import World


def command_mode(seed=0, spacing=0.4, **weights):
    weights = {BoxColor(color): weight for color, weight in weights.items()} or {color: 1.0 for color in BoxColor}
    world = World(ConveyorConfig(speed=0.1, spacing=spacing, color_weights=weights, seed=seed))
    return world, CommandMode(world, GroundTruthDetector(world).detect)


def test_timeout_when_no_box_of_the_color_arrives_and_invalid_text():
    world, mode = command_mode(red=1.0)                           # only red boxes on the belt
    result = mode.execute("pick a blue box and place it at A", timeout=10.0)
    assert not result.success and result.failed_stage == "timeout" and result.robot_id is None
    assert result.time == pytest.approx(10.0)
    assert world.C.getJointState() == pytest.approx(World().C.getJointState())    # nobody moved
    with pytest.raises(ValueError, match="no destination"):
        mode.execute("pick a blue box")
