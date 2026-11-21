"""Autonomous mode: the robots sort the boxes coming down the belt.

    python scripts/run_autonomous.py --view                                   two coordinated robots
    python scripts/run_autonomous.py --mode single --view                     Robot 1 alone
    python scripts/run_autonomous.py --mode fixed --speed 0.15 --seed 3 --view
    python scripts/run_autonomous.py --scene clear-and-sort --view            a gray box blocks a red one
    python scripts/run_autonomous.py --scene covered --view                   a red box under a cover (#177)
    python scripts/run_autonomous.py --covered 0.5 --speed 0.05 --view        half the colored boxes under a cover
    python scripts/run_autonomous.py --detector yolo --duration 40 --video autonomous.mp4     (headless: xvfb-run -a ...)

Modes: single (Robot 1 sorts red / green / blue into regions A / B / C, gray boxes leave the belt), fixed (Robot 2
also removes gray boxes into the bin), coordinated (both may sort green, one robot at a time in the shared
area, clear-and-sort for a wanted box blocked by a gray one).
"""
import argparse

import cv2

from conveyor_sort.coordination.covers import Uncovering
from conveyor_sort.interfaces import BoxColor
from conveyor_sort.metrics import MODES
from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.covers import get_cover_detector
from conveyor_sort.perception.detectors import DETECTORS, get_detector
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.covers import CoverConfig, CoveredConveyor
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
parser.add_argument("--scene", choices=["stream", "clear-and-sort", "covered"], default="stream",
                    help="clear-and-sort: only a red box with a gray box 8 cm in front of it, belt at 0.05 m/s; "
                         "covered: only a red box under a cover, belt at 0.05 m/s")
parser.add_argument("--covered", type=float, default=0.0,
                    help="covered boxes mode (#177): share of colored boxes that come under a cover (0: off)")
parser.add_argument("--view", action="store_true", help="show the RAI viewer")
parser.add_argument("--video", metavar="PATH", help="save the top-camera view as an mp4")
args = parser.parse_args()
if (args.covered or args.scene == "covered") and args.mode == "single":
    parser.error("covered boxes need both robots (Robot 2 lifts the covers): --mode fixed or coordinated")

if args.scene == "clear-and-sort":
    world = World(ConveyorConfig(speed=0.05, max_boxes=0), physics=True)
    world.conveyor.spawn_box(BoxColor.RED, x=-0.9, y=0.0)
    world.conveyor.spawn_box(BoxColor.GRAY, x=-0.82, y=0.0)
    args.duration = min(args.duration, 45.0)
elif args.scene == "covered":            # Robot 2 lifts the cover off, Robot 1 sorts the red box (#177)
    world = World(ConveyorConfig(speed=0.05, max_boxes=0), physics=True, covers=CoverConfig())
    world.conveyor.cover(world.conveyor.spawn_box(BoxColor.RED, x=-0.9, y=0.0))
    args.duration = min(args.duration, 45.0)
else:
    world = World(ConveyorConfig(speed=args.speed, spacing=args.spacing, seed=args.seed, blocked_pairs=args.blocked_pairs),
                  physics=True, covers=CoverConfig(share=args.covered, seed=args.seed) if args.covered else None)
camera = Camera(world.C)                 # video: the synced state, rendered every step (BotOp images are 2 per s)
detect = get_detector(args.detector, world)   # on BotOp images
writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), round(1 / world.dt), (1280, 720)) if args.video else None


frames = [0]                                    # video frames written so far


def on_step(world: World) -> None:
    if args.view:
        world.C.view(False, f"{args.mode}, t = {world.time:.2f} s")
    if writer:
        frame = cv2.cvtColor(camera.capture(world.time).rgb, cv2.COLOR_RGB2BGR)
        due = round(world.time / world.dt)       # physics steps are longer than dt: repeat the frame so
        for _ in range(due - frames[0]):          # that the video plays in real time
            writer.write(frame)
        frames[0] = max(frames[0], due)


roles, clear_and_sort = MODES[args.mode]
uncovering = None
if isinstance(world.conveyor, CoveredConveyor):   # covered boxes mode (#177): Robot 2 lifts the covers
    uncovering = Uncovering(world, get_cover_detector(args.detector, world))
stats = run_robots(world, detect, args.duration, roles, on_step, clear_and_sort, covers=uncovering)
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
if uncovering:
    s = uncovering.stats
    print(f"covers: {s.lifted} lifted, {s.put_aside} put aside, {s.handed_over} boxes handed over to robot 1, "
          f"{s.infeasible} too late, {s.lost} lost, failures {dict(s.failures)}, "
          f"{len(s.cover_contacts)} contacts of robot 1 with a cover")
print(f"arms: closest approach {stats.min_arm_distance * 100:.1f} cm, {len(stats.arm_contacts)} contacts")
world.close()
