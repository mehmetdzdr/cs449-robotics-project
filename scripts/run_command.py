"""Command mode: execute user instructions on the running conveyor.

    python scripts/run_command.py "pick a blue box and place it at B" "put a gray box into the bin" --view
    python scripts/run_command.py --view                    type commands interactively, empty line to quit
    xvfb-run -a python scripts/run_command.py "pick a red box and place it at A" --video command.mp4
"""
import argparse

import cv2

from conveyor_sort.command.mode import CommandMode
from conveyor_sort.command.parser import parse_command
from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.hsv import HsvDetector
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.world import World

parser = argparse.ArgumentParser()
parser.add_argument("commands", nargs="*", help='e.g. "pick a blue box and place it at B"; none: read from the keyboard')
parser.add_argument("--detector", choices=["gt", "hsv"], default="gt", help="ground truth or HSV vision")
parser.add_argument("--speed", type=float, default=0.1, help="belt speed [m/s]")
parser.add_argument("--spacing", type=float, default=0.4, help="distance between boxes [m]")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--timeout", type=float, default=60.0, help="how long to wait for a feasible box [s]")
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


def typed_commands():
    while text := input("command> ").strip():
        yield text


mode = CommandMode(world, detect, on_step)
for text in args.commands or typed_commands():
    try:
        command = parse_command(text)
    except ValueError as error:
        print(f"rejected: {error}")
        continue
    result = mode.execute(command, args.timeout)
    outcome = "done" if result.correct else f"FAILED ({result.failed_stage})"
    print(f"'{text}': {outcome}, robot {result.robot_id}, {result.time:.1f} s")
if writer:
    writer.release()
    print(f"saved {args.video}")
