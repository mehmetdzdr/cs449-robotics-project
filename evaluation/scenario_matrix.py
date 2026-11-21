"""Scenario matrix (#106): belt speed x lateral position x neighbourhood, with the expected outcome of every cell.

    python evaluation/scenario_matrix.py --jobs 6      (physics, real time; headless: xvfb-run -a ...; about 10 min)

Every cell is one BotOp physics run with ground-truth detections: a colored target and its gray neighbours, Robot 1
sorting the target and Robot 2 removing gray blockers (`ROLES`). Writes results/scenario_matrix.csv (one row per
cell: expected and observed outcome, arm contacts, a box left in a gripper); `evaluation/make_figures.py` turns it
into a table. tests/scenarios/test_matrix.py checks the planning decisions of every cell (fast) and runs the same
cells on physics (slow, nightly).
"""
import argparse
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from conveyor_sort.coordination.allocation import COLORED, LEAD_TIME, MAX_PICK_SPEED, MIN_TRACK_AGE
from conveyor_sort.perception.detectors import get_detector
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.scenarios import LATERAL, make_matrix_scenario
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots

SPEEDS = (0.05, 0.1, 0.2, 0.3)                                  # belt speed [m/s]
LAYOUTS = ("free", "front", "back", "front_back", "left_right", "front_side", "surrounded")
ROLES = {1: COLORED, 2: ()}      # Robot 1 sorts the target; Robot 2 has no color of its own and only removes blockers


def valid_laterals(layout: str) -> list[str]:
    """The lateral positions (LATERAL) where `layout` with touching neighbours fits on the belt."""
    result = []
    for lateral in LATERAL:
        try:
            make_matrix_scenario(layout, lateral=lateral)
            result.append(lateral)
        except ValueError:
            pass
    return result


def cells() -> list[tuple[str, str, float]]:
    return [(layout, lateral, speed) for layout in LAYOUTS for lateral in valid_laterals(layout) for speed in SPEEDS]


def expected(layout: str, speed: float) -> str:
    """The outcome the cell must have.

    - "sorted": a grasp axis is free (along the belt: free, left_right; across: front, back, front_back).
    - "cleared": front_side blocks both axes, one gray box is removed by Robot 2 (12.5 s), then the target sorted;
      only at 0.05 m/s, faster the target leaves Robot 1's reach before.
    - "infeasible": reported, no arm moves (surrounded needs two removals, 25 s; front_side above 0.05 m/s).
    - "missed" at 0.3 m/s: boxes faster than allocation.MAX_PICK_SPEED (0.25 m/s) are not picked (at 0.3 m/s a grasp at
      the far edge near the middle of the belt let Robot 1's place motion touch Robot 2 resting at home).
    """
    if speed > MAX_PICK_SPEED:
        return "missed"
    if layout == "surrounded" or (layout == "front_side" and speed > 0.05):
        return "infeasible"
    if layout == "front_side":
        return "cleared"
    return "sorted"


def decision_x(speed: float) -> float:
    """Target position for the planning-only check (tests): grasped at x = -0.2 (Robot 1's zone) LEAD_TIME later."""
    return -0.50 if speed == 0.05 else -0.20 - speed * LEAD_TIME


def spawn_x(speed: float) -> float:
    """Where the target starts in the physics run: tracked for MIN_TRACK_AGE and planned LEAD_TIME ahead, it is
    grasped at about x = -0.2; the slowest belt starts just upstream of Robot 1's reach, the faster ones at most at
    x = -1.0 (a neighbour behind the target must still be on the belt, which starts at -1.1)."""
    if speed <= 0.05:
        return -0.55
    return max(-0.20 - speed * (MIN_TRACK_AGE + LEAD_TIME), -1.0)


def observed(stats, target_color) -> str:
    target_sorted = any(e.correct and e.color == target_color for e in stats.events)
    if target_sorted:        # a grasp across the belt is a clearing job too, but without a removal
        return "cleared" if any(e.color != target_color and e.correct for e in stats.events) else "sorted"
    if stats.infeasible and not stats.events:
        return "infeasible"
    if not stats.events:
        return "missed"
    return "failed"


def matches(expected_outcome: str, observed_outcome: str) -> bool:
    return observed_outcome in expected_outcome.split(" or ")


def run_cell(cell: tuple[str, str, float]) -> dict:
    """One physics run of a cell; returns its record."""
    layout, lateral, speed = cell
    outcome = expected(layout, speed)
    scenario = make_matrix_scenario(layout, seed=0, lateral=lateral, x=spawn_x(speed))
    world = World(ConveyorConfig(speed=speed, max_boxes=0), physics=True)
    try:
        centre_y = world.conveyor.geometry.y
        world.conveyor.spawn_box(scenario.target_color, x=scenario.target[0], y=scenario.target[1] - centre_y)
        for color, position in scenario.neighbours:
            world.conveyor.spawn_box(color, x=position[0], y=position[1] - centre_y)
        stats = run_robots(world, get_detector("gt", world), 50.0 if outcome == "cleared" else 25.0, ROLES,
                           clear_and_sort=True)
        fingers = world.C.getJointState()[list(world.fingers.values())]
        held = bool(world.conveyor.held) or not bool(np.all(fingers > 0.035))
        result = observed(stats, scenario.target_color)
        return {"layout": layout, "lateral": lateral, "speed": speed, "expected": outcome, "observed": result,
                "ok": matches(outcome, result) and not stats.arm_contacts and not held,
                "failed_stages": " ".join(e.failed_stage for e in stats.events if e.failed_stage),
                "arm_contacts": len(stats.arm_contacts), "min_arm_distance": stats.min_arm_distance,
                "box_held": held}
    finally:
        world.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1, help="runs at the same time (separate processes)")
    parser.add_argument("--out", type=Path, default=Path("results/scenario_matrix.csv"))
    args = parser.parse_args()
    if args.jobs == 1:
        records = [run_cell(c) for c in cells()]
    else:
        with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            records = list(pool.map(run_cell, cells()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"{sum(r['ok'] for r in records)} of {len(records)} cells as expected; results in {args.out}")
    for r in records:
        if not r["ok"]:
            print(f"  {r['layout']} / {r['lateral']} / {r['speed']}: expected {r['expected']}, observed {r['observed']}"
                  f" ({r['failed_stages'] or '-'}), contacts {r['arm_contacts']}, box held {r['box_held']}")


if __name__ == "__main__":
    main()
