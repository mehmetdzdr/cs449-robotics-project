"""Cover detection (#177): where the covers on the belt are, as `Detection`s of their knob (the grasp point).

Covers are detected apart from the boxes (their own detector and tracker, `coordination.covers`), so the box
detectors and the interfaces stay as they are. The box under a cover is not seen by any box detector: the camera
cannot see it, and the ground-truth detector leaves it out (`CoveredConveyor.on_belt`).

- "gt": the covers standing on the belt, from the simulation.
- vision ("hsv", and also for "yolo" / "sam" / "sam_b": they were not trained on covers): gray pixels (the HSV
  range of gray boxes) in blobs of a cover's size whose top is at the height of a knob. A gray box is a quarter of
  that size and lower, the box detectors in turn reject a cover by its size and height (`hsv.MAX_AREA`,
  `hsv.HEIGHT_TOLERANCE`).
"""
from typing import Callable

import cv2
import numpy as np

from conveyor_sort.interfaces import BoxColor, Detection
from conveyor_sort.perception.camera import Camera, Observation
from conveyor_sort.perception.hsv import HEIGHT_TOLERANCE, HSV_RANGES, belt_roi
from conveyor_sort.perception.localization import mask_to_world
from conveyor_sort.sim.conveyor import BeltGeometry
from conveyor_sort.sim.covers import COVER_KNOB, COVER_REST
from conveyor_sort.sim.world import World

# blob size [px]: a cover is about 55 x 55 px from the top camera (2 mm per px at its height, side faces add some)
MIN_AREA, MAX_AREA = 2000, 4500
MIN_FILL = 0.75   # blob area / bounding box area: a cover is a square seen from above (arm parts are not)
CLOSE = 7         # [px] gaps closed before the blobs are found: the knob's rounded edges render darker (as box edges do,
#                   see the HSV notes in CLAUDE.md) and would cut it off from the plate around it


class HsvCoverDetector:
    def __init__(self, belt: BeltGeometry):
        self.belt = belt

    def detect(self, obs: Observation) -> list[Detection]:
        rows, cols = belt_roi(obs, self.belt)
        hsv = cv2.cvtColor(obs.rgb[rows, cols], cv2.COLOR_RGB2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for low, high in HSV_RANGES[BoxColor.GRAY]:
            mask |= cv2.inRange(hsv, np.array(low), np.array(high))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((CLOSE, CLOSE), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        detections = []
        for label in range(1, n):
            area, w, h = stats[label, cv2.CC_STAT_AREA], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
            if not (MIN_AREA <= area <= MAX_AREA and area >= MIN_FILL * w * h):
                continue
            full = np.zeros(obs.depth.shape, bool)
            full[rows, cols] = labels == label
            knob = mask_to_world(full, obs, object_height=COVER_KNOB)   # the top face is the knob's
            if abs(knob[2] - (self.belt.z_top + COVER_REST)) < HEIGHT_TOLERANCE:
                detections.append(Detection(BoxColor.GRAY, knob, 1.0, obs.timestamp))
        return detections


def get_cover_detector(name: str, world: World, camera: Camera | None = None) -> Callable[[], list[Detection]]:
    """`detect()`: the covers seen now (knob centres), for the box detector `name` (see the module docstring)."""
    if name == "gt":
        def detect_gt() -> list[Detection]:
            return [Detection(BoxColor.GRAY, world.C.getFrame(c).getPosition(), 1.0, world.time)
                    for c in world.conveyor.covers_on_belt()]
        return detect_gt
    detector = HsvCoverDetector(world.conveyor.geometry)
    camera = camera or Camera(world.C, bot=world.bot, segmentation=False)
    last = [None]

    def detect() -> list[Detection]:
        obs = camera.capture(world.time)
        if camera.bot is not None and (obs.timestamp == last[0] or obs.timestamp == float("-inf")):   # as in
            return []                                                     # perception.detectors.get_detector
        last[0] = obs.timestamp
        return detector.detect(obs)
    return detect
