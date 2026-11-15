import pytest

from conveyor_sort.command.mode import CommandMode
from conveyor_sort.interfaces import BoxColor, Command
from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.hsv import HsvDetector
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.regions import region_contains
from conveyor_sort.sim.world import World


def command_mode(seed=0, spacing=0.4, **weights):
    weights = {BoxColor(color): weight for color, weight in weights.items()} or {color: 1.0 for color in BoxColor}
    world = World(ConveyorConfig(speed=0.1, spacing=spacing, color_weights=weights, seed=seed))
    return world, CommandMode(world, GroundTruthDetector(world).detect)


@pytest.mark.parametrize("text, color, region, robot", [
    ("pick a blue box and place it at B", BoxColor.BLUE, "region_B", None),      # either robot
    ("pick a red box and place it at C", BoxColor.RED, "region_C", 1),
    ("put a green box into the bin", BoxColor.GREEN, "bin_gray", 2),
    ("move a gray box to region A", BoxColor.GRAY, "region_A", 1),
])
def test_command_is_carried_out(text, color, region, robot):
    world, mode = command_mode()
    collisions = []

    def check(world):
        world.C.computeCollisions()
        collisions.extend(world.C.getCollisions(0.0))

    mode.on_step = check
    result = mode.execute(text, timeout=90.0)
    assert result.success and result.correct and result.failed_stage is None
    assert result.command == Command(color, region)
    assert world.conveyor.colors[result.box] == color
    assert region_contains(world.C, region, world.C.getFrame(result.box).getPosition())
    assert world.C.getFrame(result.box).getParent().name == region
    assert robot is None or result.robot_id == robot
    assert collisions == []
    print(f"'{text}': robot {result.robot_id}, {result.time:.1f} s")


def test_several_commands_in_a_row_also_into_the_same_region():
    world, mode = command_mode(seed=1)
    results = [mode.execute(text) for text in ("pick a red box and place it at B", "pick a blue box and place it at B",
                                               "pick a green box and place it at A")]
    assert all(r.correct for r in results)
    assert results[0].box not in world.conveyor.colors            # collected when the next box came to region B
    assert world.C.getFrame(results[1].box).getParent().name == "region_B"
    assert world.C.getFrame(results[2].box).getParent().name == "region_A"


def test_timeout_when_no_box_of_the_color_arrives_and_invalid_text():
    world, mode = command_mode(red=1.0)                           # only red boxes on the belt
    result = mode.execute("pick a blue box and place it at A", timeout=10.0)
    assert not result.success and result.failed_stage == "timeout" and result.robot_id is None
    assert result.time == pytest.approx(10.0)
    assert world.C.getJointState() == pytest.approx(World().C.getJointState())    # nobody moved
    with pytest.raises(ValueError, match="no destination"):
        mode.execute("pick a blue box")


def test_command_with_hsv_vision():
    world = World(ConveyorConfig(speed=0.1, spacing=100.0, color_weights={BoxColor.GREEN: 1.0}, seed=0))   # one green box
    camera, detector = Camera(world.C), HsvDetector(world.conveyor.geometry)
    mode = CommandMode(world, lambda: detector.detect(camera.capture(world.time)))
    result = mode.execute("pick a green box and place it at B", timeout=30.0)
    assert result.correct and result.time < 20.0
