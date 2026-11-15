"""Prediction benchmark: constant-velocity window fit against the Kalman filter.

    python evaluation/benchmark_prediction.py

Writes results/prediction.csv: mean and 95th percentile error per predictor, noise, number of frames and
horizon. `evaluation/make_figures.py` turns it into the error-vs-horizon plot and the table.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from conveyor_sort.prediction.benchmark import prediction_errors

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=Path, default=Path("results/prediction.csv"))
parser.add_argument("--trials", type=int, default=500)
args = parser.parse_args()

HORIZONS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]       # [s]; a pick is planned 3 s ahead
NOISES = [0.0, 0.002, 0.005]                          # [m]; the detectors measured about 2 mm
FRAMES = [20, 40, 100]                                # detections seen before predicting (20 Hz)

rows = []
for frames in FRAMES:
    for predictor in ("cv", "kalman"):
        for noise in NOISES:
            errors = prediction_errors(predictor, noise, HORIZONS, frames, trials=args.trials) * 1000
            for horizon, column in zip(HORIZONS, errors.T):
                rows.append({"predictor": predictor, "noise_mm": noise * 1000, "frames": frames, "horizon_s": horizon,
                             "mean_error_mm": round(column.mean(), 3), "p95_error_mm": round(np.percentile(column, 95), 3)})

args.csv.parent.mkdir(parents=True, exist_ok=True)
with args.csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {args.csv} ({len(rows)} rows)")
