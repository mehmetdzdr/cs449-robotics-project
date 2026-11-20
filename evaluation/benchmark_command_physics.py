"""Command mode on the BotOp physics backend (#109): random commands at 0.1 and 0.2 m/s, also with blocked boxes.

    python evaluation/benchmark_command_physics.py --jobs 8            (headless: xvfb-run -a ...; about 30 min)
    python evaluation/benchmark_command_physics.py --n 10 --jobs 4     quicker

Configurations: belt 0.1 m/s (a box every 4 s) and 0.2 m/s (every 2 s), as in results/command.csv and
command_fast.csv (kinematic), each with a normal stream and one where every spawn is a gray box with a colored
box 8 cm behind it (`blocked_pairs=1.0`: every box a command can match is blocked along the belt). The commands of
a configuration are split into chunks of CHUNK, each run on its own belt (own seed) in its own process: the
physics simulation runs in real time. One row per command goes to results/command_physics.csv; per configuration
the success rate, the time per command and the failure reasons are printed.
"""
import argparse
import csv
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from conveyor_sort.command.benchmark import random_commands, run_commands

CHUNK = 5            # commands per belt
SPACING = 0.4        # distance between boxes [m]
CONFIGS = [(0.1, 0.0), (0.2, 0.0), (0.1, 1.0), (0.2, 1.0)]   # (belt speed, blocked pairs)


def chunk(args: tuple) -> list[dict]:
    speed, blocked, seed, n = args
    return run_commands(random_commands(n, seed), speed, SPACING, seed, physics=True, blocked_pairs=blocked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="commands per configuration")
    parser.add_argument("--jobs", type=int, default=8, help="belts at the same time (separate processes)")
    parser.add_argument("--out", type=Path, default=Path("results/command_physics.csv"))
    args = parser.parse_args()
    work = [(speed, blocked, seed, min(CHUNK, args.n - seed * CHUNK))
            for speed, blocked in CONFIGS for seed in range((args.n + CHUNK - 1) // CHUNK)]
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        records = [r for chunk_records in pool.map(chunk, work) for r in chunk_records]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"{len(records)} commands, results in {args.out}")
    for speed, blocked in CONFIGS:
        rs = [r for r in records if r["speed"] == speed and r["blocked_pairs"] == blocked]
        correct = [r for r in rs if r["correct_color"] and r["correct_region"]]
        times = [r["time"] for r in correct]
        reasons = Counter(r["failed_stage"] for r in rs if r not in correct)
        print(f"{speed} m/s, {'blocked boxes' if blocked else 'normal stream'}: {len(correct)}/{len(rs)} correct "
              f"({len(correct) / len(rs):.0%}), time {np.mean(times) if times else float('nan'):.1f} s mean / "
              f"{max(times, default=float('nan')):.1f} s max, failures {dict(reasons)}, "
              f"robots {dict(Counter(r['robot'] for r in rs))}")


if __name__ == "__main__":
    main()
