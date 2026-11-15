"""YOLO + SAM detector: YOLO finds the boxes, SAM turns each bounding box into a pixel mask, the mask is localized
with the depth image."""
from pathlib import Path

import numpy as np

from conveyor_sort.interfaces import BoxColor, Detection
from conveyor_sort.perception.camera import Observation
from conveyor_sort.perception.dataset import CLASSES
from conveyor_sort.perception.yolo import DEFAULT_WEIGHTS, YoloDetector, localize
from conveyor_sort.sim.conveyor import BeltGeometry

WEIGHTS_DIR = DEFAULT_WEIGHTS.parent
# name -> (weights file, downloaded by Ultralytics when missing; image size). SAM-B at 1280 px needs more than
# 8 GB of GPU memory and takes seconds per frame.
SAM_MODELS = {"mobile_sam": (WEIGHTS_DIR / "mobile_sam.pt", 1280), "sam_b": (WEIGHTS_DIR / "sam_b.pt", 1024)}


class SamDetector:
    def __init__(self, belt: BeltGeometry, model: str = "mobile_sam", yolo_weights: str | Path = DEFAULT_WEIGHTS):
        from ultralytics import SAM       # optional dependency: pip install -e ".[learning]"
        self.belt = belt
        self.yolo = YoloDetector(belt, yolo_weights)
        weights, self.image_size = SAM_MODELS[model]
        self.sam = SAM(str(weights))

    def masks(self, obs: Observation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boolean masks (N, H, W), one per YOLO box, with the YOLO class ids (N,) and confidences (N,)."""
        xyxy, classes, confidences = self.yolo.boxes(obs)
        if len(xyxy) == 0:                # SAM cannot be called without a prompt
            return np.zeros((0, *obs.depth.shape), bool), classes, confidences
        bgr = np.ascontiguousarray(obs.rgb[..., ::-1])
        result = self.sam.predict(bgr, bboxes=xyxy, imgsz=self.image_size, verbose=False)[0]
        return result.masks.data.cpu().numpy().astype(bool), classes, confidences

    def detect(self, obs: Observation) -> list[Detection]:
        detections = []
        for mask, cls, confidence in zip(*self.masks(obs)):
            position = localize(mask, obs, self.belt)
            if position is not None:
                detections.append(Detection(BoxColor(CLASSES[int(cls)]), position, float(confidence), obs.timestamp))
        return detections
