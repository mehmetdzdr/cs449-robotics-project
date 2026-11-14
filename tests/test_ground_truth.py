import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.sim.conveyor import BOX_SIZE, ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.world import World


def test_detections_match_boxes_on_belt():
    world = World(ConveyorConfig(speed=0.1, spacing=0.3))
    world.run(10.0)
    detections = GroundTruthDetector(world).detect()

    names = world.conveyor.on_belt()
    assert len(detections) == len(names) == 4
    for name, det in zip(names, detections):
        assert det.color == world.conveyor.colors[name]
        assert det.position == pytest.approx(world.C.getFrame(name).getPosition())
        assert det.position[2] == pytest.approx(world.conveyor.geometry.z_top + BOX_SIZE / 2)
        assert det.confidence == 1.0
        assert det.timestamp == pytest.approx(10.0)


def test_grasped_box_is_not_detected():
    world = World(ConveyorConfig(spacing=100.0))
    world.step()
    (name,) = world.conveyor.on_belt()
    world.conveyor.attach(name, "l_gripper")
    assert GroundTruthDetector(world).detect() == []


def test_consecutive_detections_give_belt_velocity():
    world = World(ConveyorConfig(speed=0.15, spacing=100.0, color_weights={BoxColor.RED: 1.0}))
    detector = GroundTruthDetector(world)
    world.step()
    first = detector.detect()[0]
    world.run(1.0)
    second = detector.detect()[0]
    velocity = (second.position - first.position) / (second.timestamp - first.timestamp)
    assert velocity == pytest.approx([0.15, 0.0, 0.0])
    assert isinstance(first.position, np.ndarray)
