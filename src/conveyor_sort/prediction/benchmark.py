"""Prediction benchmark: future-position error of the trackers' predictors on a box moving at constant speed."""
import numpy as np

from conveyor_sort.interfaces import BoxColor, Detection
from conveyor_sort.prediction.tracker import Tracker


def prediction_errors(predictor: str, noise: float, horizons: list[float], frames: int = 40, speed: float = 0.15,
                      rate: float = 20.0, trials: int = 200, seed: int = 0) -> np.ndarray:
    """Position error [m] of predictions made after `frames` detections, shape (trials, len(horizons)).

    One box moves along +x at `speed`; it is detected at `rate` Hz with Gaussian noise of standard deviation
    `noise` [m] in x and y. After the last detection its position `horizon` seconds later is predicted.
    """
    rng = np.random.default_rng(seed)
    velocity = np.array([speed, 0.0, 0.0])
    errors = np.empty((trials, len(horizons)))
    for trial in range(trials):
        tracker = Tracker(predictor=predictor)
        for k in range(frames):
            t = k / rate
            position = velocity * t + np.append(rng.normal(0, noise, 2), 0.65)
            (track,) = tracker.update([Detection(BoxColor.RED, position, 1.0, t)], t)
        for i, horizon in enumerate(horizons):
            errors[trial, i] = np.linalg.norm(track.predict(t + horizon) - (velocity * (t + horizon) + [0, 0, 0.65]))
    return errors
