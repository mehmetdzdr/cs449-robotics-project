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


def future_errors_mm(window: int, noise: float, horizon: float, predictor: str = "cv", frames: int = 40) -> np.ndarray:
    """Track one box moving at 0.15 m/s from noisy detections (20 Hz), predict `horizon` s ahead; 200 trials."""
    rng = np.random.default_rng(0)
    velocity = np.array([0.15, 0.0, 0.0])
    errors = []
    for _ in range(200):
        tracker = Tracker(window=window, predictor=predictor)
        for k in range(frames):
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


@pytest.mark.parametrize("predictor", ["cv", "kalman"])
def test_both_predictors_track_and_estimate_velocity(predictor):
    tracker = Tracker(predictor=predictor)
    for k in range(20):
        (track,) = tracker.update([det(BoxColor.RED, 0.01 * k, 0.2, 0.1 * k)], 0.1 * k)
    assert track.track_id == 0
    assert track.velocity == pytest.approx([0.1, 0, 0], abs=1e-3)
    assert track.predict(2.9) == pytest.approx([0.29, 0.2, 0.65], abs=1e-3)


def test_kalman_is_exact_without_noise():
    assert future_errors_mm(window=20, noise=0.0, horizon=2.0, predictor="kalman").max() < 0.1


@pytest.mark.parametrize("frames", [10, 20, 40, 100])
def test_kalman_compared_with_constant_velocity_fit(frames):
    """With few frames both use all detections and are equal; later the Kalman filter keeps using the whole
    history while the fit only sees its window of 20, so the Kalman error keeps shrinking."""
    cv = future_errors_mm(window=20, noise=0.002, horizon=2.0, frames=frames).mean()
    kalman = future_errors_mm(window=20, noise=0.002, horizon=2.0, predictor="kalman", frames=frames).mean()
    print(f"{frames} frames, mean error at 2 s: constant-velocity fit {cv:.2f} mm, Kalman {kalman:.2f} mm")
    assert kalman < cv * 1.1
    if frames >= 40:
        assert kalman < cv * 0.75


# ---------------------------------------------------------------- displaced boxes (#104)

def track_with_jump(predictor: str, jump: np.ndarray, noise: float = 0.0):
    """A box on the belt (0.1 m/s), detected every 0.05 s for 2 s, then displaced by `jump`, then 10 more steps."""
    rng = np.random.default_rng(0)
    tracker, ids = Tracker(predictor=predictor), set()
    for k in range(50):
        t = 0.05 * k
        offset = jump if k >= 40 else np.zeros(3)
        position = np.array([-0.5 + 0.1 * t, 0.2, 0.65]) + offset + np.append(rng.normal(0, noise, 2), 0)
        tracks = tracker.update([Detection(BoxColor.RED, position, 1.0, t)], t)
        ids |= {track.track_id for track in tracks if track.timestamp == t}
    return ids, tracks[-1] if len(tracks) == 1 else max(tracks, key=lambda tr: tr.timestamp)


@pytest.mark.parametrize("predictor", ["cv", "kalman"])
def test_a_box_bumped_by_2_cm_keeps_its_track_and_velocity(predictor):
    ids, track = track_with_jump(predictor, np.array([0.0, 0.02, 0.0]))
    truth = np.array([-0.5 + 0.1 * 2.45, 0.22, 0.65])
    assert len(ids) == 1                                           # followed, no new track
    assert np.linalg.norm(track.position - truth) < 0.001          # at the new place right away
    assert np.linalg.norm(track.velocity - [0.1, 0, 0]) < 0.005    # the belt still carries it
    assert np.linalg.norm(track.predict(4.45) - (truth + [0.2, 0, 0])) < 0.005


@pytest.mark.parametrize("predictor", ["cv", "kalman"])
def test_a_larger_jump_starts_a_new_track(predictor):
    ids, _ = track_with_jump(predictor, np.array([0.0, 0.08, 0.0]))   # beyond the 5 cm gate
    assert len(ids) == 2


@pytest.mark.parametrize("predictor", ["cv", "kalman"])
def test_detection_noise_is_no_jump(predictor):
    """2 mm noise without a displacement: the velocity estimate stays good (no history shifted)."""
    ids, track = track_with_jump(predictor, np.zeros(3), noise=0.002)
    assert len(ids) == 1 and np.linalg.norm(track.velocity - [0.1, 0, 0]) < 0.02
