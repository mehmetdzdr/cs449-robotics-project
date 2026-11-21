"""Experiments: run the sorting loop with parameters from a config file and log the metrics of the run."""
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from conveyor_sort.coordination.allocation import COLORED, FIXED_ROLES
from conveyor_sort.interfaces import BoxColor
from conveyor_sort.perception.detectors import get_detector
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import SortingStats, run_robots

FINISH_TIME = 12.0   # time for the pick-and-place of the last box, added to the automatic duration [s]
# manipulation configurations: name -> (roles, clear-and-sort)
MODES = {"single": ({1: COLORED}, False),          # Robot 1 alone: gray boxes leave the belt
         "fixed": (FIXED_ROLES, False),            # Robot 1 sorts, Robot 2 removes gray boxes
         "coordinated": (FIXED_ROLES, True)}       # fixed roles and clear-and-sort for blocked boxes (until #154
                                                    # also both robots sorted green, into the shared region B)


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    speed: float = 0.1                    # belt speed [m/s]
    spacing: float = 0.6                  # distance between consecutive boxes [m]
    boxes: int = 10                       # number of boxes sent down the belt
    color_weights: dict[str, float] = field(default_factory=lambda: {c.value: 1.0 for c in BoxColor})
    seed: int = 0
    detector: str = "gt"                  # one of perception.detectors.DETECTORS
    duration: float | None = None         # simulation time [s]; None: until the last box has passed the belt
    mode: str = "single"                  # "single", "fixed" or "coordinated" (see MODES)
    blocked_pairs: float = 0.0            # share of spawns that are a gray box right in front of a colored one


def load_config(path: str | Path) -> ExperimentConfig:
    """Read an ExperimentConfig from a JSON file; unknown keys are an error."""
    return ExperimentConfig(**json.loads(Path(path).read_text()))


def summarize(stats: SortingStats, boxes: int, duration: float) -> dict[str, float]:
    """Metrics of one run. Rates are NaN when there was nothing to count."""
    events = stats.events
    grasped = sum(e.failed_stage not in ("pick", "grasp") for e in events)
    placed = sum(e.success for e in events)
    return {
        "boxes": boxes,                                                         # boxes that entered the belt
        "attempts": len(events),
        "sorted_correctly": stats.sorted_correctly,
        "missed": stats.missed,                                                 # left the belt end unpicked
        "grasp_success_rate": grasped / len(events) if events else np.nan,      # box in the gripper / attempts
        "sorting_accuracy": stats.sorted_correctly / placed if placed else np.nan,   # in the right region / placed
        "planning_time_mean": float(np.mean([e.planning_time for e in events])) if events else np.nan,   # per attempt [s]
        "planning_time_max": max((e.planning_time for e in events), default=np.nan),
        "completion_time": max((e.time for e in events), default=np.nan),      # when the last attempt ended [s]
        "throughput": stats.sorted_correctly / duration * 60,                   # correctly sorted boxes per minute
        "sorted_fraction": stats.sorted_correctly / boxes if boxes else np.nan,  # of all boxes that entered the belt
        "conflicts": stats.conflicts,                                           # boxes refused: shared area occupied
        "waiting_time": stats.waiting_time,                                     # robot time idle for that reason [s]
        "clear_and_sort_attempts": len(stats.clear_and_sort),
        "clear_and_sort_success_rate": float(np.mean(stats.clear_and_sort)) if stats.clear_and_sort else np.nan,
        "grasp_failures": stats.failures["grasp"],                              # fingers closed on nothing
        "slips": stats.failures["slip"],                                        # box did not rise with the gripper
        "drops": stats.failures["drop"],                                        # box lost during transport
        "dropped_on": json.dumps(dict(stats.drops)),                            # where slipped / dropped boxes ended
        "other_failures": sum(n for stage, n in stats.failures.items() if stage not in ("grasp", "slip", "drop")),
        "failures": json.dumps(dict(stats.failures)),                           # failed attempts per stage
        "tracking_error_mean": float(np.mean(stats.tracking_errors)) if stats.tracking_errors else np.nan,   # [m]
        "tracking_error_max": max(stats.tracking_errors, default=np.nan),
        "min_arm_distance": stats.min_arm_distance,                             # physics backend [m]
        "arm_contacts": len(stats.arm_contacts),
    }


def run_experiment(config: ExperimentConfig, on_step: Callable[[World], None] | None = None) -> dict:
    """Run one experiment; returns the config values and the metrics as one flat record."""
    weights = {BoxColor(color): weight for color, weight in config.color_weights.items()}
    if config.mode not in MODES:
        raise ValueError(f"unknown mode '{config.mode}'")
    roles, clear_and_sort = MODES[config.mode]
    world = World(ConveyorConfig(config.speed, config.spacing, color_weights=weights, seed=config.seed,
                                 max_boxes=config.boxes, blocked_pairs=config.blocked_pairs), physics=True)
    try:
        detect = get_detector(config.detector, world)
        belt = world.conveyor.geometry
        duration = config.duration or ((config.boxes - 1) * config.spacing + belt.x_end - belt.x_start) / config.speed + FINISH_TIME
        stats = run_robots(world, detect, duration, roles, on_step, clear_and_sort)
        record = asdict(config) | {"duration": duration} | summarize(stats, world.conveyor.spawned, duration)
        record["color_weights"] = json.dumps(config.color_weights)   # one CSV cell
        record["blocked_pairs_spawned"] = world.conveyor.pairs
        return record
    finally:
        world.close()


def append_csv(path: str | Path, record: dict) -> None:
    """Append the record as one row; the header is written when the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        if new:
            writer.writeheader()
        writer.writerow(record)
