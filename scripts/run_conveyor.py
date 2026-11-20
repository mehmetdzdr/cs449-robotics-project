"""Run the conveyor without robots: boxes spawn, travel along the belt and fall off the end.

Examples:
    python scripts/run_conveyor.py --view
    xvfb-run -a python scripts/run_conveyor.py --duration 20 --video conveyor.mp4   (headless)
"""
import argparse

import cv2

from conveyor_sort.perception.camera import Camera
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=30.0, help="simulation time [s]")
parser.add_argument("--speed", type=float, default=0.1, help="belt speed [m/s]")
parser.add_argument("--spacing", type=float, default=0.3, help="distance between boxes [m]")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--view", action="store_true", help="show the RAI viewer")
parser.add_argument("--video", metavar="PATH", help="save the top-camera view as an mp4")
args = parser.parse_args()

world = World(ConveyorConfig(speed=args.speed, spacing=args.spacing, seed=args.seed), physics=True)
camera = writer = None
if args.video:
    camera = Camera(world.C)
    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), round(1 / world.dt), (1280, 720))


frames = [0]                                    # video frames written so far


def on_step(world: World) -> None:
    if args.view:
        world.C.view(False, f"t = {world.time:.2f} s")
    if writer:
        frame = cv2.cvtColor(camera.capture(world.time).rgb, cv2.COLOR_RGB2BGR)
        due = round(world.time / world.dt)       # physics steps are longer than dt: repeat the frame so
        for _ in range(due - frames[0]):          # that the video plays in real time
            writer.write(frame)
        frames[0] = max(frames[0], due)


world.run(args.duration, on_step)
if writer:
    writer.release()
    print(f"saved {args.video}")
print(f"t = {world.time:.1f} s, boxes on belt: {len(world.conveyor.on_belt())}, missed: {world.conveyor.missed}")
world.close()
