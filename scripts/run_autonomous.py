"""Autonomous mode: the robots sort the boxes coming down the belt.

    python scripts/run_autonomous.py --view                                   two coordinated robots
    python scripts/run_autonomous.py --mode single --view                     Robot 1 alone
    python scripts/run_autonomous.py --mode fixed --speed 0.15 --seed 3 --view
    python scripts/run_autonomous.py --scene clear-and-sort --view            a gray box blocks a red one
    python scripts/run_autonomous.py --detector yolo --duration 40 --video autonomous.mp4     (headless: xvfb-run -a ...)

Modes: single (Robot 1 sorts red / green / blue into regions A / B / C, gray boxes leave the belt), fixed (Robot 2
also removes gray boxes into the bin), coordinated (both may sort green, one robot at a time in the shared
area, clear-and-sort for a wanted box blocked by a gray one).
"""
import argparse

import cv2

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.metrics import MODES
from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.detectors import DETECTORS, get_detector
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=list(MODES), default="coordinated")
parser.add_argument("--detector", choices=DETECTORS, default="gt", help="ground truth, HSV, YOLO, YOLO + (Mobile)SAM")
parser.add_argument("--speed", type=float, default=0.1, help="belt speed [m/s]")
parser.add_argument("--spacing", type=float, default=0.6, help="distance between boxes [m]")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--duration", type=float, default=60.0, help="simulation time [s]")
parser.add_argument("--blocked-pairs", type=float, default=0.0, help="share of spawns that are a gray box right in front of a colored one")
parser.add_argument("--scene", choices=["stream", "clear-and-sort"], default="stream",
                    help="clear-and-sort: only a red box with a gray box 8 cm in front of it, belt at 0.05 m/s")
parser.add_argument("--view", action="store_true", help="show the RAI viewer")
parser.add_argument("--video", metavar="PATH", help="save the top-camera view as an mp4")
args = parser.parse_args()

if args.scene == "clear-and-sort":
    world = World(ConveyorConfig(speed=0.05, max_boxes=0))
    world.conveyor.spawn_box(BoxColor.RED, x=-0.9, y=0.0)
    world.conveyor.spawn_box(BoxColor.GRAY, x=-0.82, y=0.0)
    args.duration = min(args.duration, 45.0)
else:
    world = World(ConveyorConfig(speed=args.speed, spacing=args.spacing, seed=args.seed, blocked_pairs=args.blocked_pairs))
camera = Camera(world.C)
detect = get_detector(args.detector, world, camera)
writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), round(1 / world.dt), (1280, 720)) if args.video else None


def on_step(world: World) -> None:
    if args.view:
        world.C.view(False, f"{args.mode}, t = {world.time:.2f} s")
    if writer:
        writer.write(cv2.cvtColor(camera.capture(world.time).rgb, cv2.COLOR_RGB2BGR))


roles, clear_and_sort = MODES[args.mode]
stats = run_robots(world, detect, args.duration, roles, on_step, clear_and_sort)
if writer:
    writer.release()
    print(f"saved {args.video}")
for e in stats.events:
    outcome = "ok" if e.correct else f"FAILED at {e.failed_stage}"
    print(f"t = {e.time:6.1f} s  robot {e.robot_id}  {e.color.value:5s} -> {e.region}: {outcome}  (planning {e.planning_time:.2f} s)")
print(f"sorted correctly: {stats.sorted_correctly}, failed: {stats.failed}, left the belt: {stats.missed}")
if len(roles) == 2:
    print(f"shared area: {stats.conflicts} conflicts, {stats.waiting_time:.1f} s waiting")
if stats.clear_and_sort:
    print(f"clear-and-sort: {sum(stats.clear_and_sort)} of {len(stats.clear_and_sort)} succeeded")
