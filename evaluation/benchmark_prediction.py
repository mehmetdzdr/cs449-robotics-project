"""Prediction benchmark: constant-velocity window fit against the Kalman filter.

    python evaluation/benchmark_prediction.py            (needs matplotlib: pip install -e ".[eval]")

Writes results/prediction.csv (mean and 95th percentile error per predictor, noise, number of frames and
horizon) and the error-vs-horizon plot evaluation/figures/prediction_error.png.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conveyor_sort.prediction.benchmark import prediction_errors

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=Path, default=Path("results/prediction.csv"))
parser.add_argument("--figure", type=Path, default=Path(__file__).parent / "figures" / "prediction_error.png")
parser.add_argument("--trials", type=int, default=500)
args = parser.parse_args()

HORIZONS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]       # [s]; a pick is planned 3 s ahead
NOISES = [0.0, 0.002, 0.005]                          # [m]; the detectors measured about 2 mm
FRAMES = [20, 40, 100]                                # detections seen before predicting (20 Hz)
NAMES = {"cv": "constant-velocity fit (window 20)", "kalman": "Kalman filter"}

rows = []
figure, axes = plt.subplots(1, len(FRAMES), figsize=(4.5 * len(FRAMES), 3.8), sharey=True)
for axis, frames in zip(axes, FRAMES):
    for predictor, style in (("cv", "o-"), ("kalman", "s--")):
        for noise, color in zip(NOISES, ("tab:gray", "tab:blue", "tab:red")):
            errors = prediction_errors(predictor, noise, HORIZONS, frames, trials=args.trials) * 1000
            for horizon, column in zip(HORIZONS, errors.T):
                rows.append({"predictor": predictor, "noise_mm": noise * 1000, "frames": frames, "horizon_s": horizon,
                             "mean_error_mm": round(column.mean(), 3), "p95_error_mm": round(np.percentile(column, 95), 3)})
            axis.plot(HORIZONS, errors.mean(axis=0), style, color=color, markersize=4,
                      label=f"{NAMES[predictor]}, noise {noise * 1000:.0f} mm")
    axis.set_title(f"{frames} detections ({frames / 20:.0f} s of tracking)")
    axis.set_xlabel("prediction horizon [s]")
    axis.grid(alpha=0.3)
axes[0].set_ylabel("mean position error [mm]")
axes[0].legend(fontsize=7)
figure.tight_layout()
args.figure.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(args.figure, dpi=150)

args.csv.parent.mkdir(parents=True, exist_ok=True)
with args.csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {args.csv} ({len(rows)} rows) and {args.figure}")

print("mean error [mm] at a 3 s horizon      " + "".join(f"{f:>4d} frames: fit / Kalman   " for f in FRAMES))
for noise in NOISES:
    cells = []
    for frames in FRAMES:
        fit, kalman = (next(r["mean_error_mm"] for r in rows if (r["predictor"], r["noise_mm"], r["frames"], r["horizon_s"])
                            == (p, noise * 1000, frames, 3.0)) for p in ("cv", "kalman"))
        cells.append(f"{fit:11.2f} / {kalman:<6.2f}       ")
    print(f"noise {noise * 1000:.0f} mm" + " " * 28 + "".join(cells))
