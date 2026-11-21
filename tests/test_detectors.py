import numpy as np
import pytest

from conveyor_sort.metrics import ExperimentConfig, run_experiment
from conveyor_sort.perception.detectors import DETECTORS, get_detector
from conveyor_sort.perception.yolo import DEFAULT_WEIGHTS
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World

LEARNED = pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="fine-tuned weights are not in git (see README)")


@pytest.mark.parametrize("name", ["gt", "hsv", pytest.param("yolo", marks=LEARNED), pytest.param("sam", marks=LEARNED)])
def test_every_detector_sees_the_same_boxes(name):
    assert name in DETECTORS
    world = World(ConveyorConfig(speed=0.2, spacing=0.3, seed=4))
    world.run(6.0)
    detections = get_detector(name, world)()
    truth = {n: world.C.getFrame(n).getPosition() for n in world.conveyor.on_belt()}
    assert len(detections) == len(truth) == 5
    for d in detections:
        assert d.timestamp == pytest.approx(world.time)
        assert any(world.conveyor.colors[n] == d.color and np.linalg.norm(d.position - p) < 0.005 for n, p in truth.items())


def test_unknown_detector_is_an_error():
    with pytest.raises(ValueError, match="unknown detector 'yolo9000': choose one of gt, hsv, yolo, sam, sam_b"):
        get_detector("yolo9000", World())
    with pytest.raises(ValueError, match="unknown detector"):
        run_experiment(ExperimentConfig(detector="yolo9000"))
