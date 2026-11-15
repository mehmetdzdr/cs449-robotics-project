"""Detector selection by name, so that the pipeline switches detectors without code changes."""
from typing import Callable

from conveyor_sort.interfaces import Detection
from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.hsv import HsvDetector
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.world import World

DETECTORS = ("gt", "hsv", "yolo", "sam", "sam_b")   # ground truth, HSV thresholds, YOLO, YOLO + MobileSAM, YOLO + SAM-B


def get_detector(name: str, world: World, camera: Camera | None = None) -> Callable[[], list[Detection]]:
    """`detect()` for the sorting loop or command mode: the boxes seen now, as `Detection`s.

    The vision detectors look through `camera` (default: the top camera). YOLO and SAM need the optional
    dependencies (`pip install -e ".[learning]"`) and the fine-tuned weights (see README).
    """
    if name == "gt":
        return GroundTruthDetector(world).detect
    belt = world.conveyor.geometry
    if name == "hsv":
        detector = HsvDetector(belt)
    elif name == "yolo":
        from conveyor_sort.perception.yolo import YoloDetector
        detector = YoloDetector(belt)
    elif name in ("sam", "sam_b"):
        from conveyor_sort.perception.sam import SamDetector
        detector = SamDetector(belt, "mobile_sam" if name == "sam" else "sam_b")
    else:
        raise ValueError(f"unknown detector '{name}': choose one of {', '.join(DETECTORS)}")
    camera = camera or Camera(world.C)
    return lambda: detector.detect(camera.capture(world.time))
