"""Prediction benchmark: future-position error of the trackers' predictors.

`prediction_errors`: a synthetic box moving at constant speed, detected at a fixed rate.
`record_run` + `recorded_errors`: boxes on the physical belt (`World(physics=True)`), detected at the irregular
steps of the free-running physics clock; both predictors are evaluated on the same recording.
"""
from dataclasses import dataclass
from typing import Callable

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


@dataclass
class Frame:
    """One step of a recorded run."""
    time: float
    detections: list[Detection]
    boxes: dict[str, np.ndarray]   # ground truth: centre of every box on the belt, by name


def record_run(world, detect: Callable[[], list[Detection]], duration: float) -> list[Frame]:
    """Run `world` for `duration` seconds and record, after every step, the detections and the true box centres.

    Boxes are read from `conveyor.colors` (not `on_belt()`, which leaves out boxes near a gripper with physics),
    up to the belt end: a box past it is tipping over and no longer moves with the belt.
    """
    frames = []
    x_end = world.conveyor.geometry.x_end

    def on_step(world):
        boxes = {name: world.C.getFrame(name).getPosition().copy() for name in world.conveyor.colors}
        frames.append(Frame(world.time, detect(), {name: p for name, p in boxes.items() if p[0] <= x_end}))

    world.run(duration, on_step)
    return frames


def _true_position(frames: list[Frame], times: np.ndarray, box: str, t: float) -> np.ndarray | None:
    """Centre of `box` at time `t`, interpolated linearly between the two recorded steps around `t`."""
    k = int(np.searchsorted(times, t))
    if k == 0 or k == len(frames) or box not in frames[k - 1].boxes or box not in frames[k].boxes:
        return None   # outside the recording, or the box was not on the belt
    a, b = frames[k - 1], frames[k]
    s = (t - a.time) / (b.time - a.time)
    return (1 - s) * a.boxes[box] + s * b.boxes[box]


def recorded_errors(frames: list[Frame], predictor: str, horizons: list[float], history: int = 20,
                    max_match: float = 0.03) -> list[np.ndarray]:
    """Position errors [m] per horizon of predictions made on a recording, one array per horizon.

    The tracker sees the recorded detections step by step. From every step at which a track has at least
    `history` detections, its position `horizon` seconds later is predicted and compared with the true position
    of the box it follows (the nearest box within `max_match`; a track without one is a false detection and
    skipped). Predictions whose target time is past the recording, or after the box has left the belt, are skipped.
    """
    times = np.array([f.time for f in frames])
    tracker = Tracker(predictor=predictor)
    seen: dict[int, int] = {}   # track id -> detections so far
    errors: list[list[float]] = [[] for _ in horizons]
    for frame in frames:
        for track in tracker.update(frame.detections, frame.time):
            if abs(track.timestamp - frame.time) > 1e-9:
                continue   # not detected in this step
            seen[track.track_id] = seen.get(track.track_id, 0) + 1
            if seen[track.track_id] < history or not frame.boxes:
                continue
            box = min(frame.boxes, key=lambda name: np.linalg.norm(frame.boxes[name] - track.position))
            if np.linalg.norm(frame.boxes[box] - track.position) > max_match:
                continue
            for i, horizon in enumerate(horizons):
                target = _true_position(frames, times, box, frame.time + horizon)
                if target is not None:
                    errors[i].append(float(np.linalg.norm(track.predict(frame.time + horizon) - target)))
    return [np.array(e) for e in errors]
