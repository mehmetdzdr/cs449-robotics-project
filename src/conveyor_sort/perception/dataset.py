"""Synthetic YOLO dataset: random belt scenes rendered by the top camera, labels from ground-truth segmentation."""
from pathlib import Path

import cv2
import numpy as np

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.perception.camera import Camera, Observation
from conveyor_sort.sim.conveyor import BOX_SIZE, ConveyorConfig
from conveyor_sort.sim.scene import HOME_Q
from conveyor_sort.sim.world import World

CLASSES = [c.value for c in BoxColor]   # class id = index: red, green, blue, gray
MIN_VISIBLE_PIXELS = 150                # boxes hidden by an arm beyond this are not labelled (full box is about 900 px)
REACH_Q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 2.0, -0.5])   # arm stretched over the belt


def randomize_scene(world: World, rng: np.random.Generator, max_boxes: int = 12) -> None:
    """Replace all boxes by a random arrangement on the belt and move both arms to random poses."""
    conveyor = world.conveyor
    for name in list(conveyor.colors):
        world.C.delFrame(world.C.getFrame(name))
    conveyor.colors.clear()

    belt = conveyor.geometry
    xs: list[float] = []
    for _ in range(rng.integers(0, max_boxes + 1)):
        x = rng.uniform(belt.x_start + BOX_SIZE, belt.x_end - BOX_SIZE)
        if all(abs(x - other) > 1.5 * BOX_SIZE for other in xs):   # no overlapping boxes
            xs.append(x)
            frame = world.C.getFrame(conveyor.spawn_box())
            rel = frame.getRelativePosition()
            frame.setRelativePosition([x - (belt.x_start + belt.x_end) / 2, rel[1], rel[2]])

    arms = []
    for _ in range(2):   # each arm: somewhere between retracted and stretched over the belt, random base rotation
        q = HOME_Q + rng.uniform(0, 1) * (REACH_Q - HOME_Q)
        q[0] = rng.uniform(-0.8, 0.8)
        arms.append(q)
    world.C.setJointState(np.concatenate(arms))


def yolo_labels(world: World, obs: Observation) -> list[str]:
    """One line per visible box: `class x_center y_center width height`, normalized to [0, 1]."""
    height, width = obs.depth.shape
    lines = []
    for name in world.conveyor.on_belt():
        vs, us = np.nonzero(obs.frame_mask(world.C, name))
        if len(us) < MIN_VISIBLE_PIXELS:
            continue
        u0, u1, v0, v1 = us.min(), us.max() + 1, vs.min(), vs.max() + 1
        cls = CLASSES.index(world.conveyor.colors[name].value)
        lines.append(f"{cls} {(u0 + u1) / 2 / width:.6f} {(v0 + v1) / 2 / height:.6f} "
                     f"{(u1 - u0) / width:.6f} {(v1 - v0) / height:.6f}")
    return lines


def generate(out_dir: Path, n_images: int, val_fraction: float = 0.2, seed: int = 0) -> None:
    """Write images/{train,val}, labels/{train,val} and data.yaml in Ultralytics YOLO layout."""
    rng = np.random.default_rng(seed)
    world = World(ConveyorConfig(seed=seed))
    camera = Camera(world.C)
    n_val = round(n_images * val_fraction)
    for i in range(n_images):
        split = "val" if i < n_val else "train"
        for kind in ("images", "labels"):
            (out_dir / kind / split).mkdir(parents=True, exist_ok=True)
        randomize_scene(world, rng)
        obs = camera.capture()
        cv2.imwrite(str(out_dir / "images" / split / f"{i:05d}.png"), cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR))
        (out_dir / "labels" / split / f"{i:05d}.txt").write_text("\n".join(yolo_labels(world, obs)) + "\n")

    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
