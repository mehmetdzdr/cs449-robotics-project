"""3D localization: pixel mask + depth image -> object position in the world frame.

Shared by every detector (HSV, YOLO, SAM): they only have to produce a mask.
"""
import numpy as np

from conveyor_sort.perception.camera import Observation
from conveyor_sort.sim.conveyor import BOX_SIZE


def pixels_to_world(us: np.ndarray, vs: np.ndarray, depths: np.ndarray, obs: Observation) -> np.ndarray:
    """Back-project pixels (u = column, v = row) with their depths to world points, shape (N, 3).

    Pinhole model in the camera frame:  x = (u - c_x) d / f_x,  y = (v - c_y) d / f_y,  z = d
    then  p_world = R p_cam + t.
    """
    fx, fy, cx, cy = obs.fxycxy
    cam = np.stack([(us - cx) * depths / fx, (vs - cy) * depths / fy, depths], axis=1)
    return cam @ obs.cam_rotation.T + obs.cam_position


def mask_to_world(mask: np.ndarray, obs: Observation, object_height: float = BOX_SIZE) -> np.ndarray | None:
    """Centre of the object under `mask`, in world coordinates; None if the mask is empty.

    Only the top face is used: away from the image centre the mask also contains side faces,
    which would pull a plain centroid outwards. The centre is half the object height below the top face.
    """
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        return None
    points = pixels_to_world(us, vs, obs.depth[vs, us], obs)
    z_top = np.percentile(points[:, 2], 90)
    top = points[points[:, 2] > z_top - 0.005]
    centre = top.mean(axis=0)
    centre[2] = z_top - object_height / 2
    return centre
