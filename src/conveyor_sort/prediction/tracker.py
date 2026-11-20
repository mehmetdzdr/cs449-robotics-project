"""Multi-object tracker: associates detections over time into tracks with stable IDs and
estimates each track's velocity with a constant-velocity model: a least-squares fit over a window
of recent detections (`predictor="cv"`) or a Kalman filter (`predictor="kalman"`)."""
from dataclasses import dataclass, field

import numpy as np

from conveyor_sort.interfaces import BoxColor, Detection, Track


@dataclass
class _TrackState:
    track_id: int
    color: BoxColor
    times: list[float] = field(default_factory=list)          # timestamps of matched detections
    positions: list[np.ndarray] = field(default_factory=list)  # their positions

    window: int = 20                                           # number of recent detections used for the fit

    def add(self, time: float, position: np.ndarray) -> None:
        self.times.append(time)
        self.positions.append(position)

    def to_track(self) -> Track:
        """Constant-velocity model p(t) = p0 + v t, least-squares fitted to the last `window` detections."""
        t = np.array(self.times[-self.window:])
        p = np.array(self.positions[-self.window:])
        if len(t) < 2:
            return Track(self.track_id, self.color, p[-1], np.zeros(3), t[-1])
        dt = t - t.mean()
        velocity = dt @ (p - p.mean(axis=0)) / (dt @ dt)
        position = p.mean(axis=0) + velocity * dt[-1]   # fitted position at the latest detection
        return Track(self.track_id, self.color, position, velocity, t[-1])


MEASUREMENT_STD = 0.002    # Kalman: detection noise per axis [m]
ACCELERATION_STD = 0.01    # Kalman: how much the box velocity may drift [m/s^2]; small, the belt speed is constant
INITIAL_SPEED_STD = 1.0    # Kalman: the velocity of a new track is unknown [m/s]


class _KalmanState:
    """Constant-velocity Kalman filter, state (position, velocity). The three axes are independent and share
    one 2 x 2 covariance."""

    def __init__(self, track_id: int, color: BoxColor):
        self.track_id, self.color = track_id, color
        self.times: list[float] = []
        self.position, self.velocity = np.zeros(3), np.zeros(3)
        self.P = np.diag([MEASUREMENT_STD**2, INITIAL_SPEED_STD**2])

    def add(self, time: float, position: np.ndarray) -> None:
        if not self.times:
            self.position = np.array(position, dtype=float)
        else:
            dt = time - self.times[-1]
            F = np.array([[1.0, dt], [0.0, 1.0]])
            Q = ACCELERATION_STD**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
            self.P = F @ self.P @ F.T + Q                                   # predict
            residual = position - (self.position + self.velocity * dt)
            gain = self.P[:, 0] / (self.P[0, 0] + MEASUREMENT_STD**2)        # update
            self.position = self.position + self.velocity * dt + gain[0] * residual
            self.velocity = self.velocity + gain[1] * residual
            self.P = self.P - np.outer(gain, self.P[0])
        self.times.append(time)

    def to_track(self) -> Track:
        return Track(self.track_id, self.color, self.position, self.velocity, self.times[-1])


class Tracker:
    def __init__(self, max_distance: float = 0.05, max_age: float = 2.0, window: int = 20, predictor: str = "cv"):
        assert predictor in ("cv", "kalman")
        self.predictor = predictor
        self.max_distance = max_distance   # gate: a detection further than this from a track's prediction starts a new track [m]
        self.max_age = max_age             # a track not seen for this long is dropped (e.g. box hidden by an arm, or gone) [s]
        self.window = window               # number of recent detections used for the velocity fit ("cv" only)
        self._states: list[_TrackState] = []
        self._next_id = 0

    def update(self, detections: list[Detection], time: float) -> list[Track]:
        """Feed the detections of one camera frame taken at `time`; returns all live tracks."""
        # all (track, detection) pairs of equal color within the gate, closest first
        pairs = []
        for i, state in enumerate(self._states):
            track = state.to_track()
            for j, det in enumerate(detections):    # predicted to when the image was taken (BotOp's lags, #94)
                distance = np.linalg.norm(det.position - track.predict(det.timestamp))
                if det.color == state.color and distance <= self.max_distance:
                    pairs.append((distance, i, j))
        matched_tracks, matched_dets = set(), set()
        for _, i, j in sorted(pairs):
            if i not in matched_tracks and j not in matched_dets:
                matched_tracks.add(i)
                matched_dets.add(j)
                self._states[i].add(detections[j].timestamp, detections[j].position)

        for j, det in enumerate(detections):   # unmatched detections are new boxes
            if j not in matched_dets:
                state = (_KalmanState(self._next_id, det.color) if self.predictor == "kalman"
                         else _TrackState(self._next_id, det.color, window=self.window))
                state.add(det.timestamp, det.position)
                self._states.append(state)
                self._next_id += 1

        self._states = [s for s in self._states if time - s.times[-1] <= self.max_age]
        return [s.to_track() for s in self._states]
