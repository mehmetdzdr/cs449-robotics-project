"""Multi-object tracker: associates detections over time into tracks with stable IDs."""
from dataclasses import dataclass, field

import numpy as np

from conveyor_sort.interfaces import BoxColor, Detection, Track


@dataclass
class _TrackState:
    track_id: int
    color: BoxColor
    times: list[float] = field(default_factory=list)          # timestamps of matched detections
    positions: list[np.ndarray] = field(default_factory=list)  # their positions

    def velocity(self) -> np.ndarray:
        if len(self.times) < 2:
            return np.zeros(3)
        return (self.positions[-1] - self.positions[-2]) / (self.times[-1] - self.times[-2])

    def to_track(self) -> Track:
        return Track(self.track_id, self.color, self.positions[-1], self.velocity(), self.times[-1])


class Tracker:
    def __init__(self, max_distance: float = 0.05, max_age: float = 2.0):
        self.max_distance = max_distance   # gate: a detection further than this from a track's prediction starts a new track [m]
        self.max_age = max_age             # a track not seen for this long is dropped (e.g. box hidden by an arm, or gone) [s]
        self._states: list[_TrackState] = []
        self._next_id = 0

    def update(self, detections: list[Detection], time: float) -> list[Track]:
        """Feed the detections of one camera frame taken at `time`; returns all live tracks."""
        # all (track, detection) pairs of equal color within the gate, closest first
        pairs = []
        for i, state in enumerate(self._states):
            predicted = state.to_track().predict(time)
            for j, det in enumerate(detections):
                distance = np.linalg.norm(det.position - predicted)
                if det.color == state.color and distance <= self.max_distance:
                    pairs.append((distance, i, j))
        matched_tracks, matched_dets = set(), set()
        for _, i, j in sorted(pairs):
            if i not in matched_tracks and j not in matched_dets:
                matched_tracks.add(i)
                matched_dets.add(j)
                self._states[i].times.append(detections[j].timestamp)
                self._states[i].positions.append(detections[j].position)

        for j, det in enumerate(detections):   # unmatched detections are new boxes
            if j not in matched_dets:
                self._states.append(_TrackState(self._next_id, det.color, [det.timestamp], [det.position]))
                self._next_id += 1

        self._states = [s for s in self._states if time - s.times[-1] <= self.max_age]
        return [s.to_track() for s in self._states]
