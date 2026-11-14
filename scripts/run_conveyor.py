"""Run the conveyor without robots: boxes spawn, travel along the belt and fall off the end.

Examples:
    python scripts/run_conveyor.py --view
    xvfb-run -a python scripts/run_conveyor.py --duration 20 --video conveyor.mp4   (headless)
"""
import argparse

import cv2
import robotic as ry

from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.scene import TOP_CAMERA
from conveyor_sort.sim.world import World

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=30.0, help="simulation time [s]")
parser.add_argument("--speed", type=float, default=0.1, help="belt speed [m/s]")
parser.add_argument("--spacing", type=float, default=0.3, help="distance between boxes [m]")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--view", action="store_true", help="show the RAI viewer")
parser.add_argument("--video", metavar="PATH", help="save the top-camera view as an mp4")
args = parser.parse_args()

world = World(ConveyorConfig(speed=args.speed, spacing=args.spacing, seed=args.seed))
camera = writer = None
if args.video:
    camera = ry.CameraView(world.C)
    camera.setCamera(world.C.getFrame(TOP_CAMERA))
    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), round(1 / world.dt), (1280, 720))


def on_step(world: World) -> None:
    if args.view:
        world.C.view(False, f"t = {world.time:.2f} s")
    if writer:
        rgb, _ = camera.computeImageAndDepth(world.C)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


world.run(args.duration, on_step)
if writer:
    writer.release()
    print(f"saved {args.video}")
print(f"t = {world.time:.1f} s, boxes on belt: {len(world.conveyor.on_belt())}, missed: {world.conveyor.missed}")
