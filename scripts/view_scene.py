"""Show the scene in the RAI viewer, or save the top-camera image with --save PATH (works headless)."""
import argparse

import cv2
import robotic as ry

from conveyor_sort.sim.scene import TOP_CAMERA, load_scene

parser = argparse.ArgumentParser()
parser.add_argument("--save", metavar="PATH", help="write the top-camera image instead of opening the viewer")
args = parser.parse_args()

C = load_scene()
if args.save:
    cam = ry.CameraView(C)
    cam.setCamera(C.getFrame(TOP_CAMERA))
    rgb, _ = cam.computeImageAndDepth(C)
    cv2.imwrite(args.save, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"saved {args.save}")
else:
    C.view(True)
