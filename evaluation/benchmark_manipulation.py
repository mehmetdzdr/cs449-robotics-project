"""Manipulation benchmark: one robot, two robots with fixed roles, two coordinated robots.

    python evaluation/benchmark_manipulation.py                  (headless: xvfb-run -a ...; about 20 min)
    python evaluation/benchmark_manipulation.py --quick          a small subset, about 2 min
    python evaluation/benchmark_manipulation.py --physics --jobs 4      on BotOp physics (#107): real time, about
                                                                        5 h of simulation, 4 runs at once

Every configuration runs on the same box streams (ground-truth detections): belt speeds, box counts, spacings
and color distributions, plus streams with blocked pairs (a gray box right in front of a colored one) for
clear-and-sort. One row per run goes to results/manipulation.csv (physics: results/manipulation_physics.csv);
a summary per configuration is printed.
"""
import argparse
import itertools
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from conveyor_sort.metrics import MODES, ExperimentConfig, append_csv, run_experiment

COLORS = {"uniform": {"red": 1, "green": 1, "blue": 1, "gray": 1},
          "gray-heavy": {"red": 1, "green": 1, "blue": 1, "gray": 3},
          "green-heavy": {"red": 1, "green": 3, "blue": 1, "gray": 1}}


def configs(quick: bool, physics: bool) -> list[ExperimentConfig]:
    if quick:
        streams = [(0.1, 10, 0.45, "uniform", 0.0), (0.05, 10, 0.6, "uniform", 0.5)]
    else:
        streams = [(speed, boxes, spacing, colors, 0.0) for speed, boxes, spacing, colors
                   in itertools.product((0.05, 0.1, 0.15), (10, 20), (0.3, 0.6), COLORS)]
        streams += [(speed, 20, 0.6, "uniform", 0.5) for speed in (0.03, 0.05, 0.08)]      # blocked pairs
    return [ExperimentConfig(f"{mode}/v{speed}/n{boxes}/s{spacing}/{colors}" + ("/pairs" if pairs else ""),
                             speed, spacing, boxes, COLORS[colors], seed=0, mode=mode, blocked_pairs=pairs,
                             physics=physics)
            for speed, boxes, spacing, colors, pairs in streams for mode in MODES]


def report(record: dict) -> None:
    physics = (f"  grasp/slip/drop {record['grasp_failures']}/{record['slips']}/{record['drops']}"
               f"  tracking {record['tracking_error_mean'] * 1000:.1f} mm" if record["physics"] else "")
    print(f"{record['name']:55s} sorted {record['sorted_correctly']:2d}/{record['boxes']:2d}  missed {record['missed']:2d}  "
          f"grasp {record['grasp_success_rate']:.2f}  conflicts {record['conflicts']}  "
          f"clear-and-sort {record['clear_and_sort_attempts']}{physics}", flush=True)


def summary(records: list[dict]) -> None:
    for title, with_pairs in (("streams without blocked pairs", False), ("streams with blocked pairs", True)):
        groups = defaultdict(list)
        for record in records:
            if bool(record["blocked_pairs"]) == with_pairs:
                groups[record["mode"]].append(record)
        if not groups:
            continue
        print(title)
        print(f"{'configuration':12s} {'runs':>4s} {'sorted':>7s} {'missed':>7s} {'grasp':>6s} {'accuracy':>8s} {'boxes/min':>9s} "
              f"{'planning':>9s} {'conflicts':>9s} {'waiting':>8s} {'clear-and-sort':>15s} {'grasp/slip/drop':>16s} {'tracking':>9s}")
        for mode in MODES:
            runs = groups.get(mode, [])
            if not runs:
                continue
            total = sum(r["boxes"] for r in runs)
            attempts = sum(r["clear_and_sort_attempts"] for r in runs)
            successes = sum(round(r["clear_and_sort_success_rate"] * r["clear_and_sort_attempts"]) for r in runs if r["clear_and_sort_attempts"])
            spawned = sum(r["blocked_pairs_spawned"] for r in runs)
            clear = f"{successes}/{attempts} of {spawned}" if with_pairs else "-"
            failures = "/".join(str(sum(r[k] for r in runs)) for k in ("grasp_failures", "slips", "drops"))
            print(f"{mode:12s} {len(runs):4d} {sum(r['sorted_correctly'] for r in runs) / total:7.1%} "
                  f"{sum(r['missed'] for r in runs) / total:7.1%} {np.nanmean([r['grasp_success_rate'] for r in runs]):6.2f} "
                  f"{np.nanmean([r['sorting_accuracy'] for r in runs]):8.2f} {np.mean([r['throughput'] for r in runs]):9.2f} "
                  f"{np.nanmean([r['planning_time_mean'] for r in runs]):8.2f}s {np.mean([r['conflicts'] for r in runs]):9.1f} "
                  f"{np.mean([r['waiting_time'] for r in runs]):7.1f}s {clear:>15s} {failures:>16s} "
                  f"{np.nanmean([r['tracking_error_mean'] for r in runs]) * 1000:7.1f}mm")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--physics", action="store_true", help="BotOp physics backend (real time)")
    parser.add_argument("--jobs", type=int, default=1, help="runs at the same time (separate processes)")
    args = parser.parse_args()
    out = args.out or Path("results/manipulation_physics.csv" if args.physics else "results/manipulation.csv")
    if out.exists():
        out.unlink()
    records = []
    if args.jobs == 1:
        for config in configs(args.quick, args.physics):
            records.append(run_experiment(config))
            append_csv(out, records[-1])
            report(records[-1])
    else:   # each run in its own process: BotOp's simulation runs in real time, one core each
        with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(run_experiment, c) for c in configs(args.quick, args.physics)]):
                records.append(future.result())
                append_csv(out, records[-1])
                report(records[-1])
    print(f"\nwrote {len(records)} runs to {out}\n")
    summary(records)


if __name__ == "__main__":
    main()
