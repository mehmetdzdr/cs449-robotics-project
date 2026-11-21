import numpy as np
import robotic as ry

from conveyor_sort.sim.scene import ROBOT_PREFIX, TOP_CAMERA, load_scene


def test_scene_has_two_arms_and_camera():
    C = load_scene()
    names = C.getFrameNames()
    for prefix in ROBOT_PREFIX.values():
        assert f"{prefix}gripper" in names
        assert f"{prefix}panda_base" in names
    assert TOP_CAMERA in names
    assert len(C.getJointState()) == 14  # 7 joints per arm


def test_home_pose_is_collision_free_and_retracted():
    C = load_scene()
    C.computeCollisions()
    assert C.getCollisions(0.0) == []
    for prefix in ROBOT_PREFIX.values():   # each gripper on its own robot's side, clear of the belt (y in [0.05, 0.35])
        assert abs(C.getFrame(f"{prefix}gripper").getPosition()[1] - 0.2) > 0.2


def test_top_camera_sees_whole_table():
    C = load_scene()
    cam = ry.CameraView(C)
    cam.setCamera(C.getFrame(TOP_CAMERA))
    rgb, depth = cam.computeImageAndDepth(C)
    assert rgb.shape == (720, 1280, 3)

    # project the four table-top corners into the image; all must be inside
    fx, fy, cx, cy = cam.getFxycxy()
    cam_frame = C.getFrame(TOP_CAMERA)
    R, t = cam_frame.getRotationMatrix(), cam_frame.getPosition()
    for x in (-1.15, 1.15):
        for y in (-0.62, 0.62):
            p = R.T @ (np.array([x, y, 0.6]) - t)  # world -> camera frame
            u, v = fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy
            assert 0 <= u < 1280 and 0 <= v < 720, (x, y, u, v)
