"""Perception with dense and occluded boxes (#103): HSV, YOLO and YOLO + MobileSAM on BotOp images.

    python evaluation/benchmark_perception_scenarios.py --jobs 6      (headless: xvfb-run -a ...; about 10 min)

1. Every layout of #98 (`sim.scenarios.scenarios()`, 48: neighbourhood x lateral position x spacing), on a stopped
   physics belt, one BotOp camera image each. Per layout and detector: recall (boxes found with the right color),
   wrong color, position error (`perception.benchmark.evaluate`, which skips boxes hidden by more than a third).
2. The target half covered by Robot 1's hand (the hand hovers above one side of it, at several heights): visible
   share of the target and whether each detector finds it.
Writes results/perception_scenarios.csv (one row per layout or cover and detector) and prints a summary.
"""
import argparse
import csv
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import robotic as ry

from conveyor_sort.perception.benchmark import MATCH_DISTANCE, evaluate
from conveyor_sort.perception.camera import Camera
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.scenarios import make_scenario, scenarios, spawn_scenario
from conveyor_sort.sim.world import World

TARGET_X = 0.0           # layouts are put right under the top camera [m]
DETECTORS = ("hsv", "yolo", "sam")
_detectors = {}          # per worker process: name -> detector (models are loaded once)


def detectors(belt):
    if not _detectors:
        from conveyor_sort.perception.hsv import HsvDetector
        from conveyor_sort.perception.sam import SamDetector
        from conveyor_sort.perception.yolo import YoloDetector
        _detectors.update(hsv=HsvDetector(belt), yolo=YoloDetector(belt), sam=SamDetector(belt, "mobile_sam"))
    return _detectors


def new_image(world, camera):
    """The next new BotOp camera image of the current state (with the ground-truth segmentation)."""
    first = camera.capture(world.time).timestamp
    while True:
        world.step()
        obs = camera.capture(world.time)
        if obs.timestamp != first and obs.timestamp > -np.inf:
            return obs


def hover(world, robot: int, position, duration: float = 2.5):
    """Move the robot's gripper to `position` (pointing down) and wait there."""
    from conveyor_sort.planning.pick import _komo, _solve   # arm joints only, fingers kept (physics backend)
    gripper = "l_gripper" if robot == 1 else "r_gripper"
    komo = _komo(world.C, 20)
    komo.addControlObjective([], 2, 1e0)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
    komo.addObjective([1.0], ry.FS.position, [gripper], ry.OT.eq, [1e1], position)
    komo.addObjective([0.8, 1.0], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], [0, 0, 1])
    plan = _solve(komo)
    world.arms.set_path(robot, plan.path, world.time + world._t0 + duration)
    world.run(duration + 0.5)


def layout_rows(scenario) -> list[dict]:
    world = World(ConveyorConfig(speed=0.0, max_boxes=0), physics=True)
    try:
        spawn_scenario(world, make_scenario(scenario.name, 0, scenario.lateral, scenario.spacing, x=TARGET_X))
        world.run(0.5)
        obs = new_image(world, Camera(world.C, bot=world.bot))
        rows = []
        for name, detector in detectors(world.conveyor.geometry).items():
            metrics = evaluate([(world, obs)], detector.detect)
            rows.append({"case": "layout", "layout": scenario.name, "lateral": scenario.lateral,
                         "spacing": scenario.spacing, "cover": "", "detector": name} | metrics)
        return rows
    finally:
        world.close()


def cover_rows(height: float, offset: float) -> list[dict]:
    """The target alone, Robot 1's gripper `height` above its top and `offset` beside its centre (across the belt)."""
    world = World(ConveyorConfig(speed=0.0, max_boxes=0), physics=True)
    try:
        target, _ = spawn_scenario(world, make_scenario("free", 0, x=-0.3))
        world.run(0.5)
        camera = Camera(world.C, bot=world.bot)
        free_pixels = new_image(world, camera).frame_mask(world.C, target).sum()
        position = world.C.getFrame(target).getPosition()
        hover(world, 1, position + [0, offset, 0.03 + height])
        obs = new_image(world, camera)
        visible = obs.frame_mask(world.C, target).sum() / free_pixels
        rows = []
        for name, detector in detectors(world.conveyor.geometry).items():
            found = [d for d in detector.detect(obs) if np.linalg.norm(d.position - position) < MATCH_DISTANCE]
            right = [d for d in found if d.color == world.conveyor.colors[target]]
            rows.append({"case": "cover", "layout": "free", "lateral": "", "spacing": "",
                         "cover": f"{height:.2f}/{offset:.2f}", "detector": name, "visible": round(float(visible), 3),
                         "found": bool(right), "wrong_color": bool(found) and not right,
                         "error_mean_mm": 1000 * float(np.linalg.norm(right[0].position - position)) if right else np.nan})
        return rows
    finally:
        world.close()


def run(job):
    return layout_rows(job) if not isinstance(job, tuple) else cover_rows(*job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("results/perception_scenarios.csv"))
    args = parser.parse_args()
    covers = [(h, o) for h in (0.05, 0.15) for o in (0.0, 0.03, 0.05)]
    jobs = scenarios() + covers
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = [row for job_rows in pool.map(run, jobs) for row in job_rows]
    keys = sorted({k for row in rows for k in row}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows in {args.out}")
    groups = defaultdict(list)
    for row in rows:
        if row["case"] == "layout":
            groups[(row["layout"], row["spacing"], row["detector"])].append(row)
    print(f"{'layout':12s} {'spacing':12s} {'detector':8s} {'boxes':>5s} {'recall':>7s} {'wrong':>6s} {'error mm':>9s}")
    for (layout, spacing, name), rs in sorted(groups.items()):
        boxes = sum(r["boxes"] for r in rs)
        recall = sum(r["recall"] * r["boxes"] for r in rs) / boxes if boxes else np.nan
        wrong = sum(r["wrong_color"] * r["boxes"] for r in rs) / boxes if boxes else np.nan
        print(f"{layout:12s} {spacing:12s} {name:8s} {boxes:5d} {recall:7.0%} {wrong:6.0%} "
              f"{np.nanmean([r['error_mean_mm'] for r in rs]):9.1f}")
    for row in rows:
        if row["case"] == "cover":
            print(f"cover {row['cover']:10s} {row['detector']:5s} visible {row['visible']:.0%} found {row['found']} "
                  f"wrong color {row['wrong_color']}")


if __name__ == "__main__":
    main()
