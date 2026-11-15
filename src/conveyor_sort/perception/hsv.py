"""HSV baseline detector: color thresholds, connected components, 3D localization from the blob mask."""
import cv2
import numpy as np

from conveyor_sort.interfaces import BoxColor, Detection
from conveyor_sort.perception.camera import Observation
from conveyor_sort.perception.localization import mask_to_world
from conveyor_sort.sim.conveyor import BOX_SIZE, BeltGeometry

# OpenCV HSV ranges (H 0..179, S and V 0..255), measured on rendered boxes; the belt has S = 0, V < 115
HSV_RANGES = {
    BoxColor.RED: [((0, 150, 60), (10, 255, 255)), ((170, 150, 60), (179, 255, 255))],
    BoxColor.GREEN: [((50, 150, 60), (70, 255, 255))],
    BoxColor.BLUE: [((105, 150, 60), (125, 255, 255))],
    BoxColor.GRAY: [((0, 0, 125), (179, 30, 190))],
}
MIN_AREA, MAX_AREA = 300, 2000   # blob size [px]; a fully visible box is about 900 px
HEIGHT_TOLERANCE = 0.015         # accepted deviation of the blob height from a box on the belt [m]


def belt_roi(obs: Observation, belt: BeltGeometry, margin: int = 15) -> tuple[slice, slice]:
    """Image rows / columns covering the belt (plus a margin), from the camera calibration."""
    fx, fy, cx, cy = obs.fxycxy
    corners = np.array([[x, y, belt.z_top] for x in (belt.x_start, belt.x_end)
                        for y in (belt.y - belt.width / 2, belt.y + belt.width / 2)])
    cam = (corners - obs.cam_position) @ obs.cam_rotation          # world -> camera frame
    us, vs = fx * cam[:, 0] / cam[:, 2] + cx, fy * cam[:, 1] / cam[:, 2] + cy
    height, width = obs.depth.shape
    return (slice(max(0, int(vs.min()) - margin), min(height, int(vs.max()) + margin)),
            slice(max(0, int(us.min()) - margin), min(width, int(us.max()) + margin)))


class HsvDetector:
    def __init__(self, belt: BeltGeometry):
        self.belt = belt

    def masks(self, obs: Observation) -> list[tuple[BoxColor, np.ndarray, np.ndarray]]:
        """(color, pixel mask, position) of every blob that passes the size and height checks."""
        rows, cols = belt_roi(obs, self.belt)
        hsv = cv2.cvtColor(obs.rgb[rows, cols], cv2.COLOR_RGB2HSV)
        blobs = []
        for color, ranges in HSV_RANGES.items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for low, high in ranges:
                mask |= cv2.inRange(hsv, np.array(low), np.array(high))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            for label in range(1, n):                               # label 0 is the background
                if MIN_AREA <= stats[label, cv2.CC_STAT_AREA] <= MAX_AREA:
                    full = np.zeros(obs.depth.shape, bool)
                    full[rows, cols] = labels == label
                    position = mask_to_world(full, obs)
                    # a box centre lies half a box above the belt; this rejects gray blobs on the robot arms
                    if abs(position[2] - (self.belt.z_top + BOX_SIZE / 2)) < HEIGHT_TOLERANCE:
                        blobs.append((color, full, position))
        return blobs

    def detect(self, obs: Observation) -> list[Detection]:
        return [Detection(color, position, 1.0, obs.timestamp) for color, _, position in self.masks(obs)]
