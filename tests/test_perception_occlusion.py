"""Perception with dense and occluded boxes (#103): the tracker keeps a box through an occlusion by an arm."""
import sys
from pathlib import Path

import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.perception.detectors import get_detector
from conveyor_sort.prediction.tracker import Tracker
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from benchmark_perception_scenarios import hover  # noqa: E402

pytestmark = pytest.mark.slow   # physics scenarios in real time: nightly / on demand in CI (#97)


def test_target_keeps_its_track_through_2_s_under_the_hand():
    """Robot 1's hand hovers 10 cm above the belt; a red box passes under it at 0.05 m/s and is not detected (HSV on
    BotOp images) for about 2 s. Its track must be the same before and after."""
    world = World(ConveyorConfig(speed=0.05, max_boxes=0), physics=True)
    try:
        hover(world, 1, [-0.45, 0.2, 0.79])                    # hand above the belt centre line
        box = world.conveyor.spawn_box(BoxColor.RED, x=-0.75, y=0.0)
        detect, tracker = get_detector("hsv", world), Tracker()
        ids, detected_at = set(), []
        while world.C.getFrame(box).getPosition()[0] < -0.2:  # until it is well past the hand
            world.step()
            detections = detect()                              # [] also for a repeated image (2 new ones per s)
            tracks = tracker.update(detections, world.time)
            for d in detections:
                truth = world.C.getFrame(box).getPosition() - [0.05 * (world.time - d.timestamp), 0, 0]
                if np.linalg.norm(d.position - truth) < 0.03:
                    detected_at.append(d.timestamp)
                    ids |= {t.track_id for t in tracks if np.linalg.norm(t.position - d.position) < 0.01}
        longest = max(np.diff(detected_at))
        print(f"{len(detected_at)} detections, longest gap {longest:.1f} s (hidden under the hand), track ids {ids}")
        assert longest >= 1.5                                  # it was hidden, then seen again
        assert len(ids) == 1                                   # and kept its track
    finally:
        world.close()
