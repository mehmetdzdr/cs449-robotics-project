"""Command-mode correctness test: random commands, checked against ground truth.

    python evaluation/benchmark_command.py --n 40          (headless: xvfb-run -a ...)

Writes one row per command to results/command.csv and prints the success rate.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from conveyor_sort.command.benchmark import random_commands, run_commands

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=40, help="number of commands")
parser.add_argument("--speed", type=float, default=0.1, help="belt speed [m/s]")
parser.add_argument("--spacing", type=float, default=0.4, help="distance between boxes [m]")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, default=Path("results/command.csv"))
args = parser.parse_args()

records = run_commands(random_commands(args.n, args.seed), args.speed, args.spacing, args.seed)
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

for record in records:
    if not (record["correct_color"] and record["correct_region"]):
        print(f"FAILED ({record['failed_stage']}): {record['command']}")
n = len(records)
print(f"{n} commands at {args.speed} m/s, a box every {args.spacing / args.speed:.1f} s; results in {args.out}")
for key in ("parsed", "success", "correct_color", "correct_region"):
    print(f"  {key:15s} {sum(r[key] for r in records):3d} / {n}  ({sum(r[key] for r in records) / n:.0%})")
correct = [r for r in records if r["correct_color"] and r["correct_region"]]
print(f"  success rate    {len(correct) / n:.1%} (requested color placed at the requested destination)")
print(f"  time per command: mean {np.mean([r['time'] for r in correct]):.1f} s, max {np.max([r['time'] for r in correct]):.1f} s")
print(f"  by robot: {dict(sorted(Counter(r['robot'] for r in records).items(), key=str))}")
print("  by destination: " + ", ".join(
    f"{region} {sum(r in correct for r in records if r['region'] == region)}/{sum(r['region'] == region for r in records)}"
    for region in sorted({r["region"] for r in records})))
