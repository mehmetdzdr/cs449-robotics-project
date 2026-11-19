"""Prediction on the physical belt (#111): a recorded run of World(physics=True)."""
import numpy as np

from conveyor_sort.prediction.benchmark import record_run, recorded_errors
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.world import World


def test_recorded_physical_run_with_ground_truth():
    world = World(ConveyorConfig(speed=0.1, spacing=0.3), physics=True)
    try:
        frames = record_run(world, GroundTruthDetector(world).detect, 5.0)   # boxes enter at 0 and 3 s
    finally:
        world.close()
    times = np.array([f.time for f in frames])
    assert len(frames) > 50 and np.all(np.diff(times) >= world.dt - 1e-6)
    assert len(frames[-1].boxes) == len(frames[-1].detections) == 2
    for predictor in ("cv", "kalman"):
        errors = recorded_errors(frames, predictor, [0.5, 1.0])
        print(predictor, [f"{e.mean() * 1000:.2f} mm ({len(e)})" for e in errors])
        assert all(len(e) > 10 for e in errors) and max(e.max() for e in errors) < 0.005
