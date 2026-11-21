"""Benchmark for blocked and surrounded targets (#108): success, infeasible-reported and failure rates per (layout,
speed) cell, and the extra time clearing costs.

    python evaluation/benchmark_clearing.py --jobs 6           (physics, real time; headless: xvfb-run -a ...)
    python evaluation/benchmark_clearing.py --runs 3 --jobs 6  a quick check

The layouts and the physics run of one case are those of the scenario matrix (#106, evaluation/scenario_matrix.py):
a colored target with touching gray neighbours, Robot 1 sorting it, Robot 2 removing gray blockers. Each of the `runs`
per cell cycles through the lateral positions where the layout fits and starts the target up to 3 cm earlier or later
along the belt (seeded), so the runs differ in where the boxes are when the camera images arrive. 0.3 m/s is left out:
boxes faster than allocation.MAX_PICK_SPEED are not picked. Writes results/clearing.csv (one row per run) and prints
a table per cell; the extra time is the time to sort the target minus that of a free target at the same speed.
"""
import argparse
import csv
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from conveyor_sort.perception.detectors import get_detector
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.scenarios import make_matrix_scenario
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots
from scenario_matrix import ROLES, expected, observed, spawn_x, valid_laterals

LAYOUTS = ("free", "front", "back", "front_back", "front_side", "surrounded")   # free: the baseline for the extra time
SPEEDS = (0.05, 0.1, 0.2)
JITTER = 0.03   # the target starts up to this much earlier or later along the belt [m]


def run_trial(task: tuple[str, float, int]) -> dict:
    layout, speed, run = task
    laterals = valid_laterals(layout)
    lateral = laterals[run % len(laterals)]
    x = spawn_x(speed) + np.random.default_rng(run).uniform(-JITTER, JITTER)
    scenario = make_matrix_scenario(layout, seed=run, lateral=lateral, x=x)
    outcome = expected(layout, speed)
    world = World(ConveyorConfig(speed=speed, max_boxes=0), physics=True)
    try:
        centre_y = world.conveyor.geometry.y
        world.conveyor.spawn_box(scenario.target_color, x=scenario.target[0], y=scenario.target[1] - centre_y)
        for color, position in scenario.neighbours:
            world.conveyor.spawn_box(color, x=position[0], y=position[1] - centre_y)
        start = world.time
        stats = run_robots(world, get_detector("gt", world), 50.0 if outcome == "cleared" else 25.0, ROLES,
                           clear_and_sort=True)
        result = observed(stats, scenario.target_color)
        done = [e.time for e in stats.events if e.correct and e.color == scenario.target_color]
        return {"layout": layout, "speed": speed, "run": run, "lateral": lateral, "expected": outcome,
                "outcome": result, "time": done[0] - start if done else float("nan"),
                "failed_stages": " ".join(e.failed_stage for e in stats.events if e.failed_stage),
                "arm_contacts": len(stats.arm_contacts), "min_arm_distance": stats.min_arm_distance}
    finally:
        world.close()


def summary(records: list[dict]) -> list[dict]:
    """One row per (layout, speed): rates and the mean extra time over a free target."""
    cells = defaultdict(list)
    for r in records:
        cells[r["layout"], float(r["speed"])].append(r)
    free = {speed: np.nanmean([float(r["time"]) for r in rs if r["outcome"] == "sorted"])
            for (layout, speed), rs in cells.items() if layout == "free"}
    rows = []
    for (layout, speed), rs in cells.items():
        n = len(rs)
        success = [r for r in rs if r["outcome"] in ("sorted", "cleared")]
        extra = [float(r["time"]) - free[speed] for r in success if speed in free]
        rows.append({"layout": layout, "speed": speed, "runs": n, "expected": rs[0]["expected"],
                     "success": len(success) / n, "infeasible": sum(r["outcome"] == "infeasible" for r in rs) / n,
                     "failed": sum(r["outcome"] in ("failed", "missed") for r in rs) / n,
                     "extra_time": float(np.mean(extra)) if extra else float("nan"),
                     "contacts": sum(int(r["arm_contacts"]) for r in rs)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30, help="runs per (layout, speed) cell")
    parser.add_argument("--jobs", type=int, default=1, help="runs at the same time (separate processes)")
    parser.add_argument("--out", type=Path, default=Path("results/clearing.csv"))
    args = parser.parse_args()
    tasks = [(layout, speed, run) for layout in LAYOUTS for speed in SPEEDS for run in range(args.runs)]
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        records = list(pool.map(run_trial, tasks))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"{len(records)} runs, results in {args.out}")
    print(f"{'layout':12s} {'speed':>5s} {'runs':>4s} {'expected':>10s} {'success':>8s} {'infeas':>7s} {'failed':>7s} "
          f"{'extra [s]':>9s} {'contacts':>8s}")
    for row in summary(records):
        print(f"{row['layout']:12s} {row['speed']:5.2f} {row['runs']:4d} {row['expected']:>10s} {row['success']:8.0%} "
              f"{row['infeasible']:7.0%} {row['failed']:7.0%} {row['extra_time']:+9.1f} {row['contacts']:8d}")


if __name__ == "__main__":
    main()
