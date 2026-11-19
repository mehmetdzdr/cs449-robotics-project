import numpy as np

from conveyor_sort.interfaces import BoxColor, Detection
from conveyor_sort.prediction.benchmark import Frame, prediction_errors, recorded_errors

HORIZONS = [0.5, 2.0, 5.0]


def irregular_times(n: int = 150, seed: int = 0) -> np.ndarray:
    """Step times like the physics clock's: 0.05 to 0.4 s apart."""
    return np.cumsum(np.random.default_rng(seed).uniform(0.05, 0.4, n))


def belt_recording(times, starts: dict[str, tuple[float, float]], speed: float = 0.1, x_end: float = np.inf,
                   noise: float = 0.0, seed: int = 0) -> list[Frame]:
    """Boxes starting at (x, y) and moving along +x at `speed`, detected at every step until they pass `x_end`."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in times:
        boxes = {name: np.array([x + speed * t, y, 0.65]) for name, (x, y) in starts.items() if x + speed * t <= x_end}
        detections = [Detection(BoxColor.RED, p + np.append(rng.normal(0, noise, 2), 0.0), 1.0, t) for p in boxes.values()]
        frames.append(Frame(t, detections, boxes))
    return frames


def test_errors_grow_with_horizon_and_noise_and_kalman_wins_with_a_long_history():
    errors = {(p, noise): prediction_errors(p, noise, HORIZONS, frames=40, trials=100)
              for p in ("cv", "kalman") for noise in (0.0, 0.002, 0.005)}
    for (predictor, noise), e in errors.items():
        assert e.shape == (100, 3)
        mean = e.mean(axis=0)
        if noise == 0.0:
            assert mean.max() < 1e-4                                   # exact without noise
        else:
            assert mean[0] < mean[1] < mean[2]                         # further ahead is harder
    for predictor in ("cv", "kalman"):                                 # more noise, more error
        assert np.all(errors[predictor, 0.002].mean(axis=0) < errors[predictor, 0.005].mean(axis=0))
    for noise in (0.002, 0.005):                                       # 40 frames: Kalman uses all, the fit its last 20
        assert np.all(errors["kalman", noise].mean(axis=0) < 0.8 * errors["cv", noise].mean(axis=0))


def test_same_seed_gives_same_result():
    assert np.array_equal(prediction_errors("cv", 0.002, HORIZONS, trials=5), prediction_errors("cv", 0.002, HORIZONS, trials=5))


# ---------------------------------------------------------------- recorded runs (physical belt)

def test_recorded_exact_constant_velocity_at_irregular_steps():
    frames = belt_recording(irregular_times(), {"box_0": (-1.0, 0.1), "box_1": (-1.3, 0.3)})
    for predictor in ("cv", "kalman"):
        errors = recorded_errors(frames, predictor, [0.5, 1.0, 3.0])
        assert all(len(e) > 100 for e in errors) and len(errors[0]) > len(errors[2])   # further ahead: fewer targets
        assert max(e.max() for e in errors) < 1e-6


def test_recorded_noise_gives_errors_that_grow_with_the_horizon():
    frames = belt_recording(irregular_times(300), {"box_0": (-1.0, 0.2)}, noise=0.002)
    for predictor in ("cv", "kalman"):
        mean = [e.mean() for e in recorded_errors(frames, predictor, [0.5, 1.0, 2.0, 3.0])]
        assert 0.0005 < mean[0] < mean[1] < mean[2] < mean[3] < 0.02


def test_no_prediction_past_the_belt_end_or_the_recording():
    times = irregular_times()
    frames = belt_recording(times, {"box_0": (-1.0, 0.2)}, x_end=0.5)
    present = [f.time for f in frames if f.boxes]
    for horizon, errors in zip([0.5, 3.0], recorded_errors(frames, "cv", [0.5, 3.0], history=20)):
        expected = sum(t + horizon <= present[-1] for t in present[19:])   # from the 20th detection on
        assert len(errors) == expected > 0


def test_false_and_missed_detections_add_no_predictions():
    times = irregular_times()
    frames = belt_recording(times, {"box_0": (-1.0, 0.2), "box_1": (-1.3, 0.2)})
    baseline = [len(e) for e in recorded_errors(frames, "cv", [0.5, 1.0])]
    for f in frames:
        f.detections = [d for d in f.detections if abs(d.position[0] - f.boxes["box_1"][0]) > 1e-9]   # box_1 is never seen
        f.detections.append(Detection(BoxColor.GREEN, np.array([0.0, 0.0, 0.65]), 1.0, f.time))      # a box that is not there
    errors = recorded_errors(frames, "cv", [0.5, 1.0])
    assert [2 * len(e) for e in errors] == baseline   # only box_0's predictions are left
    assert max(e.max() for e in errors) < 1e-6
