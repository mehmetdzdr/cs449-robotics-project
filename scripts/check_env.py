"""Verify the development environment: RAI import, scene loading, offscreen rendering.

Usage:  python scripts/check_env.py      (headless Linux: xvfb-run -a python scripts/check_env.py)
"""
import sys


def main() -> int:
    print(f"python {sys.version.split()[0]}")

    import numpy as np
    import cv2
    print(f"numpy {np.__version__}, opencv {cv2.__version__}")

    import robotic as ry  # needs libGLU on Linux
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandasTable.g"))
    print(f"robotic OK, example scene has {len(C.getFrameNames())} frames")

    cam = ry.CameraView(C)  # needs a display (or xvfb) for the OpenGL context
    cam.setCamera(C.getFrame("camera"))
    rgb, depth = cam.computeImageAndDepth(C)
    assert rgb.ndim == 3 and depth.shape == rgb.shape[:2]
    print(f"offscreen rendering OK, image {rgb.shape}")

    import conveyor_sort  # noqa: F401
    print("conveyor_sort package OK")
    print("ENVIRONMENT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
