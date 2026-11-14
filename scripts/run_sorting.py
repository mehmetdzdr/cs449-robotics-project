"""Single-robot autonomous sorting: Robot 1 sorts red / green / blue boxes into regions A / B / C.

    python scripts/run_sorting.py --view
    python scripts/run_sorting.py --detector hsv --duration 40 --video sorting.mp4     (headless: xvfb-run -a ...)
"""
import argparse

import cv2

from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.hsv import HsvDetector
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_single_robot

parser = argparse.ArgumentParser()
parser.add_argument("--detector", choices=["gt", "hsv"], default="gt", help="ground truth or HSV vision")
parser.add_argument("--duration", type=float, default=60.0, help="simulation time [s]")
parser.add_argument("--speed", type=float, default=0.1, help="belt speed [m/s]")
parser.add_argument("--spacing", type=float, default=0.6, help="distance between boxes [m]")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--view", action="store_true", help="show the RAI viewer")
parser.add_argument("--video", metavar="PATH", help="save the top-camera view as an mp4")
args = parser.parse_args()

world = World(ConveyorConfig(speed=args.speed, spacing=args.spacing, seed=args.seed))
camera = Camera(world.C)
if args.detector == "hsv":
    hsv = HsvDetector(world.conveyor.geometry)
    detect = lambda: hsv.detect(camera.capture(world.time))
else:
    detect = GroundTruthDetector(world).detect
writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), round(1 / world.dt), (1280, 720)) if args.video else None


def on_step(world: World) -> None:
    if args.view:
        world.C.view(False, f"t = {world.time:.2f} s")
    if writer:
        writer.write(cv2.cvtColor(camera.capture(world.time).rgb, cv2.COLOR_RGB2BGR))


stats = run_single_robot(world, detect, args.duration, on_step=on_step)
if writer:
    writer.release()
    print(f"saved {args.video}")
for e in stats.events:
    outcome = "ok" if e.correct else f"FAILED at {e.failed_stage}"
    print(f"t = {e.time:6.1f} s  {e.color.value:5s} -> {e.region}: {outcome}  (planning {e.planning_time:.2f} s)")
print(f"sorted correctly: {stats.sorted_correctly}, failed: {stats.failed}, left the belt: {stats.missed}")
