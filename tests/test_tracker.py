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
