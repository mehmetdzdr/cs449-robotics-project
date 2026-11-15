"""Autonomous sorting: Robot 1 sorts red / green / blue boxes into regions A / B / C; with --robots 2,
Robot 2 removes gray boxes into the bin at the same time; with --shared it also sorts green boxes (region B).

    python scripts/run_sorting.py --view
    python scripts/run_sorting.py --robots 2 --view
    python scripts/run_sorting.py --detector hsv --duration 40 --video sorting.mp4     (headless: xvfb-run -a ...)
"""
import argparse

import cv2

from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.detectors import DETECTORS, get_detector
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World
from conveyor_sort.coordination.allocation import FIXED_ROLES, SHARED_ROLES
from conveyor_sort.sorting import run_robots

parser = argparse.ArgumentParser()
parser.add_argument("--detector", choices=DETECTORS, default="gt", help="ground truth, HSV, YOLO, YOLO + (Mobile)SAM")
parser.add_argument("--robots", type=int, choices=[1, 2], default=1, help="1: Robot 1 alone, 2: fixed roles")
parser.add_argument("--shared", action="store_true", help="with --robots 2: both robots may sort green boxes")
parser.add_argument("--clear-and-sort", action="store_true", help="with --robots 2: Robot 2 removes a gray box that blocks a wanted box")
parser.add_argument("--duration", type=float, default=60.0, help="simulation time [s]")
parser.add_argument("--speed", type=float, default=0.1, help="belt speed [m/s]")
parser.add_argument("--spacing", type=float, default=0.6, help="distance between boxes [m]")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--view", action="store_true", help="show the RAI viewer")
parser.add_argument("--video", metavar="PATH", help="save the top-camera view as an mp4")
args = parser.parse_args()

world = World(ConveyorConfig(speed=args.speed, spacing=args.spacing, seed=args.seed))
camera = Camera(world.C)
detect = get_detector(args.detector, world, camera)
writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), round(1 / world.dt), (1280, 720)) if args.video else None


def on_step(world: World) -> None:
    if args.view:
        world.C.view(False, f"t = {world.time:.2f} s")
    if writer:
        writer.write(cv2.cvtColor(camera.capture(world.time).rgb, cv2.COLOR_RGB2BGR))


roles = SHARED_ROLES if args.shared else FIXED_ROLES
if args.robots == 1:
    roles = {1: roles[1]}
stats = run_robots(world, detect, args.duration, roles, on_step, args.clear_and_sort and args.robots == 2)
if writer:
    writer.release()
    print(f"saved {args.video}")
for e in stats.events:
    outcome = "ok" if e.correct else f"FAILED at {e.failed_stage}"
    print(f"t = {e.time:6.1f} s  robot {e.robot_id}  {e.color.value:5s} -> {e.region}: {outcome}  (planning {e.planning_time:.2f} s)")
print(f"sorted correctly: {stats.sorted_correctly}, failed: {stats.failed}, left the belt: {stats.missed}")
print(f"shared area: {stats.conflicts} conflicts, {stats.waiting_time:.1f} s waiting")
if stats.clear_and_sort:
    print(f"clear-and-sort: {sum(stats.clear_and_sort)} of {len(stats.clear_and_sort)} succeeded")
