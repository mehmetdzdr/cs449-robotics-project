"""YOLO detector: a fine-tuned Ultralytics model finds the boxes, depth inside each bounding box gives the 3D position."""
from pathlib import Path

import numpy as np

from conveyor_sort.interfaces import BoxColor, Detection
from conveyor_sort.perception.camera import Observation
from conveyor_sort.perception.dataset import CLASSES
from conveyor_sort.perception.localization import mask_to_world, pixels_to_world
from conveyor_sort.sim.conveyor import BOX_SIZE, BeltGeometry

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "weights" / "yolo_boxes.pt"   # not in git, see README
IMAGE_SIZE = 1280             # the camera resolution: a box is only about 30 px wide
HEIGHT_TOLERANCE = 0.015      # pixels this close to the height of a box top belong to the box [m]
MIN_TOP_PIXELS = 150          # of about 900 for a fully visible box


def localize(mask: np.ndarray, obs: Observation, belt: BeltGeometry) -> np.ndarray | None:
    """Box centre from a pixel mask that may also cover some belt, side faces or a part of an arm above the box:
    only the pixels at the height of a box top are used. None if there are too few of them."""
    vs, us = np.nonzero(mask)
    heights = pixels_to_world(us, vs, obs.depth[vs, us], obs)[:, 2]
    on_top = np.abs(heights - (belt.z_top + BOX_SIZE)) < HEIGHT_TOLERANCE
    if on_top.sum() < MIN_TOP_PIXELS:
        return None
    top = np.zeros(obs.depth.shape, bool)
    top[vs[on_top], us[on_top]] = True
    return mask_to_world(top, obs)


def boxes_to_detections(xyxy: np.ndarray, classes: np.ndarray, confidences: np.ndarray,
                        obs: Observation, belt: BeltGeometry) -> list[Detection]:
    """Localize each bounding box (pixels: u0, v0, u1, v1) with the depth image inside its rectangle."""
    detections = []
    for (u0, v0, u1, v1), cls, confidence in zip(xyxy, classes, confidences):
        mask = np.zeros(obs.depth.shape, bool)
        mask[max(0, round(v0)):round(v1), max(0, round(u0)):round(u1)] = True
        position = localize(mask, obs, belt)
        if position is not None:
            detections.append(Detection(BoxColor(CLASSES[int(cls)]), position, float(confidence), obs.timestamp))
    return detections


class YoloDetector:
    def __init__(self, belt: BeltGeometry, weights: str | Path = DEFAULT_WEIGHTS, confidence: float = 0.5):
        from ultralytics import YOLO      # optional dependency: pip install -e ".[learning]"
        self.belt = belt
        self.confidence = confidence
        self.model = YOLO(str(weights))

    def boxes(self, obs: Observation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bounding boxes (N, 4) as u0, v0, u1, v1 in pixels, class ids (N,) and confidences (N,)."""
        bgr = np.ascontiguousarray(obs.rgb[..., ::-1])       # Ultralytics expects BGR for numpy images
        boxes = self.model.predict(bgr, imgsz=IMAGE_SIZE, conf=self.confidence, verbose=False)[0].boxes
        return boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()

    def detect(self, obs: Observation) -> list[Detection]:
        return boxes_to_detections(*self.boxes(obs), obs, self.belt)
