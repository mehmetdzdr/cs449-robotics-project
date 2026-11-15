"""Perception benchmark: HSV, YOLO, YOLO + MobileSAM and YOLO + SAM-B on the same scenes.

    python evaluation/benchmark_perception.py          (needs .[learning] and the weights; headless: xvfb-run -a ...)

Conditions: random box arrangements, dense arrangements (touching boxes), darker and brighter images.
Writes results/perception.csv and prints the table.
"""
import argparse
import csv
from pathlib import Path

from conveyor_sort.perception.benchmark import evaluate, make_scenes
from conveyor_sort.perception.hsv import HsvDetector
from conveyor_sort.perception.sam import SamDetector
from conveyor_sort.perception.yolo import YoloDetector

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=40, help="scenes per condition")
parser.add_argument("--seed", type=int, default=1000, help="not 0: the training set was generated with seed 0")
parser.add_argument("--out", type=Path, default=Path("results/perception.csv"))
args = parser.parse_args()

CONDITIONS = {"random": ("random", 1.0), "dense (touching boxes)": ("dense", 1.0),
              "dark (x 0.6)": ("random", 0.6), "bright (x 1.4)": ("random", 1.4)}
rows = []
detectors = None
for condition, (arrangement, brightness) in CONDITIONS.items():
    scenes = make_scenes(args.n, args.seed, arrangement, brightness)
    if detectors is None:
        belt = scenes[0][0].conveyor.geometry
        hsv, yolo = HsvDetector(belt), YoloDetector(belt)
        mobile_sam, sam_b = SamDetector(belt, "mobile_sam"), SamDetector(belt, "sam_b")
        detectors = {
            "HSV": (hsv.detect, lambda obs: [mask for _, mask, _ in hsv.masks(obs)]),
            "YOLO": (yolo.detect, None),
            "YOLO + MobileSAM": (mobile_sam.detect, lambda obs: list(mobile_sam.masks(obs)[0])),
            "YOLO + SAM-B": (sam_b.detect, lambda obs: list(sam_b.masks(obs)[0])),
        }
    for name, (detect, masks) in detectors.items():
        row = {"condition": condition, "detector": name} | evaluate(scenes, detect, masks)
        rows.append(row)
        print(f"{condition:24s} {name:17s} boxes {row['boxes']:3d}  recall {row['recall']:.3f}  wrong color {row['wrong_color']:.3f}  "
              f"precision {row['precision']:.3f}  error {row['error_mean_mm']:.1f} / {row['error_max_mm']:.1f} mm  "
              f"mask IoU {row['mask_iou']:.3f}  {row['time_ms']:.0f} ms", flush=True)

args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {args.out}")
