"""Camera capture: renders rgb, depth and ground-truth segmentation from a camera frame of the scene."""
from dataclasses import dataclass

import numpy as np
import robotic as ry

from conveyor_sort.sim.scene import TOP_CAMERA


@dataclass
class Observation:
    """One camera frame. Input of every vision detector."""
    rgb: np.ndarray            # (H, W, 3) uint8
    depth: np.ndarray          # (H, W) float32, distance along the camera's viewing axis [m]
    segmentation: np.ndarray   # (H, W) uint32, RAI frame ID per pixel. GROUND TRUTH: labels / evaluation only
    fxycxy: np.ndarray         # (4,) intrinsics f_x, f_y, c_x, c_y [px]
    cam_position: np.ndarray   # (3,) camera position in the world frame [m]
    cam_rotation: np.ndarray   # (3, 3) camera-to-world rotation
    timestamp: float           # simulation time [s]

    def frame_mask(self, C: ry.Config, frame_name: str) -> np.ndarray:
        """Ground-truth pixel mask of a frame (valid only for the config state this image was rendered from)."""
        return self.segmentation == C.getFrame(frame_name).ID


class Camera:
    def __init__(self, C: ry.Config, frame_name: str = TOP_CAMERA):
        self.C = C
        self.frame_name = frame_name
        self._view = ry.CameraView(C)
        self._view.setCamera(C.getFrame(frame_name))

    def capture(self, timestamp: float = 0.0) -> Observation:
        rgb, depth = self._view.computeImageAndDepth(self.C)
        segmentation = self._view.computeSegmentationID()
        frame = self.C.getFrame(self.frame_name)
        return Observation(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            fxycxy=self._view.getFxycxy(),
            cam_position=frame.getPosition(),
            cam_rotation=frame.getRotationMatrix(),
            timestamp=timestamp,
        )
