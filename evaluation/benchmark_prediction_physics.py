"""Prediction on the physical belt: constant-velocity window fit against the Kalman filter on recorded runs.

    python evaluation/benchmark_prediction_physics.py          (headless: xvfb-run -a ...; about 2 to 3 min)

For each detector, one run of World(physics=True) is recorded (detections and true box centres at every step of
the free-running physics clock); both predictors are then evaluated on the same recording. Writes
results/prediction_physics.csv: mean and 95th percentile error per detector, predictor and horizon, with the
step lengths of the run. `evaluation/make_figures.py` turns it into a table.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from conveyor_sort.perception.detectors import get_detector
from conveyor_sort.prediction.benchmark import record_run, recorded_errors
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=Path, default=Path("results/prediction_physics.csv"))
parser.add_argument("--detectors", nargs="+", default=["gt", "hsv"])
parser.add_argument("--duration", type=float, default=60.0, help="simulation time per run [s]")
parser.add_argument("--speed", type=float, default=0.1)
parser.add_argument("--spacing", type=float, default=0.3)
parser.add_argument("--hyper-speed", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

HORIZONS = [0.5, 1.0, 2.0, 3.0]   # [s]; a pick is planned 3 s ahead

rows = []
for name in args.detectors:
    world = World(ConveyorConfig(speed=args.speed, spacing=args.spacing, seed=args.seed), physics=True,
                  hyper_speed=args.hyper_speed)
    try:
        frames = record_run(world, get_detector(name, world), args.duration)
    finally:
        world.close()
    steps = np.diff([f.time for f in frames])
    for predictor in ("cv", "kalman"):
        for horizon, errors in zip(HORIZONS, recorded_errors(frames, predictor, HORIZONS)):
            errors = errors * 1000
            rows.append({"detector": name, "predictor": predictor, "horizon_s": horizon, "predictions": len(errors),
                         "mean_error_mm": round(errors.mean(), 3) if len(errors) else "nan",
                         "p95_error_mm": round(np.percentile(errors, 95), 3) if len(errors) else "nan",
                         "step_mean_s": round(steps.mean(), 4), "step_max_s": round(steps.max(), 4),
                         "speed": args.speed, "hyper_speed": args.hyper_speed})
            print(f"{name:4} {predictor:6} {horizon:.1f} s: {rows[-1]['mean_error_mm']} mm mean, "
                  f"{rows[-1]['p95_error_mm']} mm p95 ({len(errors)} predictions)")
    print(f"{name}: {len(frames)} steps, step mean {steps.mean():.3f} s, max {steps.max():.3f} s")

args.csv.parent.mkdir(parents=True, exist_ok=True)
with args.csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {args.csv} ({len(rows)} rows)")
