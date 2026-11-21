"""Perception benchmark: detectors compared on the same rendered scenes against simulator ground truth."""
import time
from dataclasses import replace
from typing import Callable

import numpy as np
import robotic as ry

from conveyor_sort.interfaces import Detection
from conveyor_sort.perception.camera import Camera, Observation
from conveyor_sort.perception.dataset import randomize_scene
from conveyor_sort.sim.conveyor import BOX_SIZE, ConveyorConfig
from conveyor_sort.sim.scene import ROBOT_PREFIX
from conveyor_sort.sim.world import World

MIN_VISIBLE_PIXELS = 600      # a box hidden by an arm beyond this (full box: about 900 px) need not be found
PARTLY_HIDDEN = 300           # ... but between this and MIN_VISIBLE_PIXELS it counts for the partly-hidden recall
MATCH_DISTANCE = 0.03         # a detection this close to a box centre is that box [m]
TOUCH_GAP = 0.06              # centre distance of the touching boxes in the "dense" arrangement [m]: one box width


def make_scenes(n: int, seed: int, arrangement: str = "random", brightness: float = 1.0) -> list[tuple[World, Observation]]:
    """`n` random belt scenes with the arms partly over the belt, each with its own world.

    "dense": every second box is moved next to its predecessor, touching it, along or across the belt.
    "occluded": one arm's gripper hovers 5 to 15 cm above a random box, up to 5 cm beside its centre (#110).
    `brightness` scales the image (the renderer has no light control; this imitates darker / brighter lighting).
    """
    rng = np.random.default_rng(seed)
    scenes = []
    for _ in range(n):
        world = World(ConveyorConfig(seed=seed))
        randomize_scene(world, rng)
        if arrangement == "dense":
            names = world.conveyor.on_belt()
            for first, second in zip(names[::2], names[1::2]):
                offset = [TOUCH_GAP, 0, 0] if rng.random() < 0.5 else [0, TOUCH_GAP * np.sign(-world.C.getFrame(first).getRelativePosition()[1] or 1), 0]
                world.C.getFrame(second).setRelativePosition(world.C.getFrame(first).getRelativePosition() + offset)
        if arrangement == "occluded":
            hover_over_a_box(world, rng)
        obs = Camera(world.C).capture()
        if brightness != 1.0:
            obs = replace(obs, rgb=np.clip(obs.rgb.astype(float) * brightness, 0, 255).astype(np.uint8))
        scenes.append((world, obs))
    return scenes


def hover_over_a_box(world: World, rng: np.random.Generator) -> bool:
    """Put the gripper of the robot nearer to a random box 5 to 15 cm above it (IK); False if there is no box or
    neither robot reaches the point."""
    boxes = world.conveyor.on_belt()
    if not boxes:
        return False
    box = world.C.getFrame(boxes[rng.integers(len(boxes))]).getPosition()
    angle = rng.uniform(0, 2 * np.pi)
    point = box + [*(rng.uniform(0, 0.05) * np.array([np.cos(angle), np.sin(angle)])), BOX_SIZE / 2 + rng.uniform(0.05, 0.15)]
    robots = sorted(ROBOT_PREFIX, key=lambda r: abs(world.C.getFrame(f"{ROBOT_PREFIX[r]}panda_base").getPosition()[1] - point[1]))
    for robot in robots:
        komo = ry.KOMO(world.C, 1, 1, 0, True)
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
        komo.addObjective([], ry.FS.position, [f"{ROBOT_PREFIX[robot]}gripper"], ry.OT.eq, [1e1], point)
        komo.addObjective([], ry.FS.vectorZ, [f"{ROBOT_PREFIX[robot]}gripper"], ry.OT.eq, [1e1], [0, 0, 1])
        if ry.NLP_Solver(komo.nlp(), verbose=0).solve().feasible:
            world.C.setJointState(komo.getPath()[-1])
            return True
    return False


def evaluate(scenes: list[tuple[World, Observation]], detect: Callable[[Observation], list[Detection]],
             masks: Callable[[Observation], list[np.ndarray]] | None = None) -> dict[str, float]:
    """Metrics of one detector on the scenes.

    recall: well-visible boxes found with the right color; wrong_color: found, but with another color;
    precision: detections that lie on a box of their color; error: distance of a found box from its true centre;
    mask_iou: best overlap of the detector's masks (`masks(obs)`, optional) with the true box mask;
    partly_hidden_recall: boxes with PARTLY_HIDDEN to MIN_VISIBLE_PIXELS visible pixels (about a third to two thirds
    of the box) found with the right color (#110);
    time: seconds per `detect` call (first call excluded: model warm-up).
    """
    found = wrong_color = n_boxes = n_detections = on_a_box = hidden = hidden_found = 0
    errors, ious, seconds = [], [], []
    detect(scenes[0][1])
    for world, obs in scenes:
        start = time.perf_counter()
        detections = detect(obs)
        seconds.append(time.perf_counter() - start)
        scene_masks = masks(obs) if masks else []
        boxes = {name: (world.conveyor.colors[name], world.C.getFrame(name).getPosition()) for name in world.conveyor.on_belt()}
        for name, (color, position) in boxes.items():
            truth = obs.frame_mask(world.C, name)
            if truth.sum() < MIN_VISIBLE_PIXELS:
                if truth.sum() >= PARTLY_HIDDEN:
                    hidden += 1
                    hidden_found += any(d.color == color and np.linalg.norm(d.position - position) < MATCH_DISTANCE
                                        for d in detections)
                continue
            n_boxes += 1
            distances = [np.linalg.norm(d.position - position) for d in detections if d.color == color]
            if distances and min(distances) < MATCH_DISTANCE:
                found += 1
                errors.append(min(distances))
            elif any(np.linalg.norm(d.position - position) < MATCH_DISTANCE for d in detections):
                wrong_color += 1
            if scene_masks:
                ious.append(max((m & truth).sum() / (m | truth).sum() for m in scene_masks))
        n_detections += len(detections)
        on_a_box += sum(any(color == d.color and np.linalg.norm(d.position - position) < MATCH_DISTANCE + 0.01
                            for color, position in boxes.values()) for d in detections)
    return {
        "boxes": n_boxes, "recall": found / n_boxes, "wrong_color": wrong_color / n_boxes,
        "precision": on_a_box / n_detections if n_detections else np.nan,
        "error_mean_mm": 1000 * float(np.mean(errors)) if errors else np.nan,
        "error_max_mm": 1000 * float(np.max(errors)) if errors else np.nan,
        "mask_iou": float(np.mean(ious)) if ious else np.nan,
        "time_ms": 1000 * float(np.mean(seconds)),
        "partly_hidden": hidden, "partly_hidden_recall": hidden_found / hidden if hidden else np.nan,
    }
