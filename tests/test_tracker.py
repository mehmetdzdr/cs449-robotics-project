import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor, Detection
from conveyor_sort.prediction.tracker import Tracker
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.world import World


def det(color, x, y, t):
    return Detection(color, np.array([x, y, 0.65]), 1.0, t)


def test_ids_stay_stable_with_ground_truth_over_a_long_run():
    """Every box must keep one ID from entering to leaving the belt, with boxes entering and leaving."""
    world = World(ConveyorConfig(speed=0.2, spacing=0.12))   # dense: boxes 12 cm apart
    detector, tracker = GroundTruthDetector(world), Tracker()
    id_of_box: dict[str, int] = {}

    def on_step(world):
        tracks = tracker.update(detector.detect(), world.time)
        for name in world.conveyor.on_belt():
            position = world.C.getFrame(name).getPosition()
            (track,) = [t for t in tracks if np.allclose(t.position, position)]
            assert id_of_box.setdefault(name, track.track_id) == track.track_id   # never changes
            assert track.color == world.conveyor.colors[name]

    world.run(25.0, on_step)
    assert world.conveyor.missed > 10                               # boxes really left the belt
    assert len(set(id_of_box.values())) == len(id_of_box) > 30      # one distinct ID per box


def test_velocity_estimate():
    tracker = Tracker()
    tracker.update([det(BoxColor.RED, 0.00, 0.2, 0.0)], 0.0)
    (track,) = tracker.update([det(BoxColor.RED, 0.01, 0.2, 0.1)], 0.1)
    assert track.velocity == pytest.approx([0.1, 0, 0])
    assert track.predict(1.1) == pytest.approx([0.11, 0.2, 0.65])


def test_color_gating_keeps_adjacent_boxes_apart():
    tracker = Tracker()
    tracker.update([det(BoxColor.RED, 0.0, 0.2, 0.0)], 0.0)
    tracks = tracker.update([det(BoxColor.BLUE, 0.005, 0.2, 0.1)], 0.1)   # other color at nearly the same place
    assert sorted(t.color for t in tracks) == [BoxColor.BLUE, BoxColor.RED]


def test_track_survives_short_occlusion_and_is_dropped_after_max_age():
    tracker = Tracker(max_age=2.0)
    for k in range(5):
        tracker.update([det(BoxColor.GREEN, 0.01 * k, 0.2, 0.1 * k)], 0.1 * k)
    (hidden,) = tracker.update([], 1.4)                                    # box hidden for 1 s
    (track,) = tracker.update([det(BoxColor.GREEN, 0.15, 0.2, 1.5)], 1.5)   # reappears where predicted
    assert track.track_id == hidden.track_id
    assert tracker.update([], 3.6) == []                                    # unseen for more than max_age


def future_errors_mm(window: int, noise: float, horizon: float) -> np.ndarray:
    """Track one box moving at 0.15 m/s from noisy detections (20 Hz), predict `horizon` s ahead; 200 trials."""
    rng = np.random.default_rng(0)
    velocity = np.array([0.15, 0.0, 0.0])
    errors = []
    for _ in range(200):
        tracker = Tracker(window=window)
        for k in range(40):
            t = 0.05 * k
            position = velocity * t + np.append(rng.normal(0, noise, 2), 0.65)
            (track,) = tracker.update([Detection(BoxColor.RED, position, 1.0, t)], t)
        truth = velocity * (t + horizon) + [0, 0, 0.65]
        errors.append(np.linalg.norm(track.predict(t + horizon) - truth))
    return np.array(errors) * 1000


def test_prediction_is_exact_without_noise():
    assert future_errors_mm(window=20, noise=0.0, horizon=2.0).max() < 1e-6


@pytest.mark.parametrize("horizon, bound_mm", [(0.5, 3.0), (1.0, 5.0), (2.0, 9.0)])
def test_future_position_error_with_2mm_detection_noise(horizon, bound_mm):
    errors = future_errors_mm(window=20, noise=0.002, horizon=horizon)
    assert errors.mean() < bound_mm


def test_window_fit_beats_two_point_velocity():
    two_point = future_errors_mm(window=2, noise=0.002, horizon=2.0).mean()
    windowed = future_errors_mm(window=20, noise=0.002, horizon=2.0).mean()
    assert windowed < two_point / 10
    print(f"mean error at 2 s: two-point {two_point:.1f} mm, window 20: {windowed:.1f} mm")
