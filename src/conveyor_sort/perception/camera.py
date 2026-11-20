"""Camera capture: renders rgb, depth and ground-truth segmentation from a camera frame of the scene."""
from dataclasses import dataclass

import numpy as np
import robotic as ry

from conveyor_sort.sim.scene import TOP_CAMERA

IMAGE_LATENCY = 0.075   # BotOp: a new image shows the scene this long before it is first returned, captures every 0.05 s [s]


@dataclass
class Observation:
    """One camera frame. Input of every vision detector."""
    rgb: np.ndarray            # (H, W, 3) uint8
    depth: np.ndarray          # (H, W) float32, distance along the camera's viewing axis [m]
    segmentation: np.ndarray | None   # (H, W) uint32, RAI frame ID per pixel. GROUND TRUTH: labels / evaluation only
                                      # (None: Camera(..., segmentation=False))
    fxycxy: np.ndarray         # (4,) intrinsics f_x, f_y, c_x, c_y [px]
    cam_position: np.ndarray   # (3,) camera position in the world frame [m]
    cam_rotation: np.ndarray   # (3, 3) camera-to-world rotation
    timestamp: float           # simulation time [s]

    def frame_mask(self, C: ry.Config, frame_name: str) -> np.ndarray:
        """Ground-truth pixel mask of a frame (valid only for the config state this image was rendered from)."""
        return self.segmentation == C.getFrame(frame_name).ID


class Camera:
    """With `bot` (physics backend, `World.bot`) rgb, depth and intrinsics come from BotOp's simulated camera
    (`bot.getImageAndDepth`); the ground-truth segmentation is still rendered by a `CameraView` of C, which
    holds the simulated poses after every sync.

    BotOp's simulation renders a new camera image only every 0.35 to 0.5 s (software rendering under xvfb), and
    `getImageAndDepth` returns the latest one. `capture` notices a new image (call it every step) and stamps the
    observation with the capture time minus IMAGE_LATENCY; repeated images keep their time, the first one (there
    before the first capture, of unknown age) is stamped -inf. The true image time
    is not available: the stamp is off by about 0.03 to 0.07 s (standard deviation, measured), i.e. 3 to 14 mm
    along a belt at 0.1 to 0.2 m/s. The segmentation is of the current state, so it matches the image only on a
    stopped belt.
    """

    def __init__(self, C: ry.Config, frame_name: str = TOP_CAMERA, bot: ry.BotOp | None = None,
                 segmentation: bool = True):
        self.C = C
        self.frame_name = frame_name
        self.bot = bot
        # with `bot`, segmentation=False skips the CameraView render (0.13 s): the observation's segmentation is
        # None, and a capture every step (0.05 s) notices new images in time (#96: 0.15 s steps gave HSV grasps
        # 1 to 2 cm off along the belt, where the fingers leave 1 cm)
        self.segmentation = segmentation or bot is None
        self._last_rgb: np.ndarray | None = None
        self._image_time = 0.0
        self._view = None
        self._frame_names: list[str] = []

    def capture(self, timestamp: float = 0.0) -> Observation:
        # RAI's CameraView caches shapes and colors; after frames were added or removed it can render
        # stale colors, so the view is rebuilt whenever the set of frames changed.
        segmentation = None
        if self.segmentation:
            names = self.C.getFrameNames()
            if names != self._frame_names:
                self._view = ry.CameraView(self.C)
                self._view.setCamera(self.C.getFrame(self.frame_name))
                self._frame_names = names
            rgb, depth = self._view.computeImageAndDepth(self.C)
            segmentation = self._view.computeSegmentationID()
            fxycxy = self._view.getFxycxy()
        if self.bot is not None:
            rgb, depth = self.bot.getImageAndDepth(self.frame_name)
            fxycxy = np.asarray(self.bot.getCameraFxycxy(self.frame_name))
            if self._last_rgb is None:        # the image there was before: its age is unknown (seen: 0.2 m off)
                self._image_time, self._last_rgb = float("-inf"), rgb
            elif not np.array_equal(rgb, self._last_rgb):    # a new image
                self._image_time, self._last_rgb = timestamp - IMAGE_LATENCY, rgb
            timestamp = self._image_time
        frame = self.C.getFrame(self.frame_name)
        return Observation(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            fxycxy=fxycxy,
            cam_position=frame.getPosition(),
            cam_rotation=frame.getRotationMatrix(),
            timestamp=timestamp,
        )
