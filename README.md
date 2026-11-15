# Vision-Based Dynamic Multi-Robot Sorting on a Conveyor Belt

CS449/549 term project, Bilkent University.

Two manipulators in RAI share a moving conveyor. In **autonomous mode** Robot 1
sorts colored boxes into target regions while Robot 2 removes gray boxes. In
**command mode** a user instruction like *"pick a blue box and place it at B"*
is executed by choosing a matching box and robot.

## Pipeline

```
camera ─► perception ─► prediction ─► coordination ─► planning (KOMO) ─► joint trajectories
                                          ▲
                              command (optional)
```

All modules exchange only the types in
[`src/conveyor_sort/interfaces.py`](src/conveyor_sort/interfaces.py), so each
module can be developed and tested against simulator ground truth before the
upstream module is ready.

## Layout

| Path | Responsibility | Owner |
|---|---|---|
| `src/conveyor_sort/sim/` | RAI scene, conveyor motion, camera | S. E. Hyder |
| `src/conveyor_sort/perception/` | HSV baseline | O. E. Yakar |
| `src/conveyor_sort/perception/` | YOLO / SAM, 3D localization | M. Dizdar |
| `src/conveyor_sort/prediction/` | Constant-velocity and Kalman prediction | A. Müftüoğlu |
| `src/conveyor_sort/coordination/` | Task allocation, workspace priority, clear-and-sort | A. Müftüoğlu |
| `src/conveyor_sort/planning/` | KOMO pick-and-place on moving boxes | H. Arkca |
| `src/conveyor_sort/command/` | Command-driven mode | O. E. Yakar |
| `evaluation/` | Benchmarks and metrics | S. E. Hyder |
| `assets/` | RAI `.g` configuration files | — |
| `scripts/` | Entry points (autonomous / command mode) | — |
| `tests/` | Unit and ground-truth tests | everyone |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core: robotic (RAI), numpy, opencv
pip install -e ".[learning]"       # optional: YOLO, SAM
python scripts/check_env.py        # must end with ENVIRONMENT OK
python scripts/view_scene.py       # opens the scene in the RAI viewer
python scripts/run_conveyor.py --view   # boxes moving on the belt, no robots
python scripts/run_autonomous.py --view   # autonomous mode: two coordinated robots sort the boxes
python scripts/run_command.py "pick a blue box and place it at B" --view   # command mode
pytest
```

Requires Python 3.10 or newer.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: libGLU.so.1` (Linux) | `sudo apt install libglu1-mesa` |
| `ImportError: liblapack.so.3` or another missing `.so` (minimal Linux installs) | `sudo apt install liblapack3 libgl1 libfreetype6 libxrandr2 libxxf86vm1 libxi6` |
| `GLFW error ... DISPLAY environment variable is missing` (headless Linux, SSH, CI) | `sudo apt install xvfb`, then prefix commands with `xvfb-run -a`, e.g. `xvfb-run -a pytest` |
| `FreeType Error: Failed to load font` warning | Harmless; only text overlays in the viewer are disabled |

## Demos

Both entry points take `--speed`, `--spacing`, `--seed`, `--detector gt|hsv|yolo|sam|sam_b`, `--view` (RAI viewer)
and `--video out.mp4` (top-camera video; headless Linux: prefix with `xvfb-run -a`).

```bash
python scripts/run_autonomous.py --view                          # two coordinated robots
python scripts/run_autonomous.py --mode single --view            # Robot 1 alone: gray boxes leave the belt
python scripts/run_autonomous.py --mode fixed --speed 0.15 --view   # Robot 2 removes gray boxes
python scripts/run_autonomous.py --scene clear-and-sort --view   # a gray box blocks a red one: Robot 2 clears it first
python scripts/run_command.py "pick a blue box and place it at B" "put a gray box into the bin" --view
python scripts/run_command.py --view                             # type commands, empty line to quit
```

## Experiments

```bash
python evaluation/run_experiment.py evaluation/configs/example.json --out results/runs.csv
```

The JSON config sets belt speed, box spacing, number of boxes, color distribution, seed and detector. Each run
appends one row to the CSV: grasp success rate, sorting accuracy, missed boxes, planning time, completion time
and throughput (boxes per minute).

### Perception benchmark

```bash
python evaluation/benchmark_perception.py      # needs .[learning] and the weights; results/perception.csv
```

40 random scenes per condition with the arms partly over the belt (not the scenes YOLO was trained on),
about 180 well-visible boxes each, RTX 3070 Laptop GPU:

| condition | detector | recall | precision | position error mean / max | mask IoU | time per frame |
|---|---|---|---|---|---|---|
| random arrangement | HSV | 1.000 | 1.000 | 1.7 / 5.4 mm | 0.979 | 15 ms |
| | YOLO | 1.000 | 1.000 | 1.7 / 5.4 mm | (no mask) | 33 ms |
| | YOLO + MobileSAM | 1.000 | 1.000 | 1.8 / 5.8 mm | 0.975 | 92 ms |
| | YOLO + SAM-B | 1.000 | 1.000 | 1.7 / 5.6 mm | 0.982 | 232 ms |
| dense (touching boxes) | HSV | 1.000 | 1.000 | 1.8 / 8.2 mm | 0.983 | 15 ms |
| | YOLO | 0.988 | 1.000 | 1.9 / 9.4 mm | (no mask) | 43 ms |
| | YOLO + MobileSAM | 0.988 | 1.000 | 1.8 / 8.9 mm | 0.957 | 92 ms |
| | YOLO + SAM-B | 0.988 | 1.000 | 1.8 / 9.1 mm | 0.961 | 238 ms |
| image 40 % darker | HSV | 0.908 | 1.000 | 1.8 / 5.4 mm | 0.899 | 13 ms |
| | YOLO | 1.000 | 1.000 | 1.7 / 5.4 mm | (no mask) | 40 ms |
| | YOLO + MobileSAM | 1.000 | 1.000 | 1.9 / 5.7 mm | 0.975 | 93 ms |
| | YOLO + SAM-B | 1.000 | 1.000 | 1.7 / 5.6 mm | 0.981 | 234 ms |
| image 40 % brighter | HSV | 0.919 | 1.000 | 1.9 / 11.6 mm | 0.908 | 14 ms |
| | YOLO | 1.000 | 1.000 | 1.7 / 5.4 mm | (no mask) | 44 ms |
| | YOLO + MobileSAM | 1.000 | 1.000 | 1.8 / 5.7 mm | 0.972 | 95 ms |
| | YOLO + SAM-B | 1.000 | 1.000 | 1.7 / 5.6 mm | 0.983 | 235 ms |

No detector reported a wrong color. The renderer has no light control: "darker" and "brighter" scale the image.
HSV is the fastest and as accurate as the learned detectors under the lighting it was tuned for, but loses
8 to 9 % of the boxes when the brightness changes. YOLO is unaffected. SAM adds masks but no accuracy,
for 3 to 7 times the time.

### Command-mode correctness

```bash
python evaluation/benchmark_command.py --n 40      # results/command.csv
```

Random commands (color, destination and phrasing) are executed one after the other on a running belt and
checked on the simulator state. At 0.1 m/s (a box every 4 s): 40 of 40 commands ended with a box of the
requested color in the requested destination, 17.6 s per command on average (54 s at most, waiting for a box of
the color), 29 by Robot 1 and 11 by Robot 2. At 0.2 m/s (a box every 2 s, other seed): 30 of 30, 15.0 s on average.
### Manipulation benchmark

```bash
python evaluation/benchmark_manipulation.py      # 117 runs, about 20 min; --quick for a 2 min subset
```

Three configurations on the same box streams (ground-truth detections): `single` (Robot 1 alone), `fixed`
(Robot 1 sorts, Robot 2 removes gray boxes), `coordinated` (both may sort green, shared-area priority,
clear-and-sort). 36 streams: speed 0.05 / 0.1 / 0.15 m/s, 10 / 20 boxes, spacing 0.3 / 0.6 m, three color
distributions. Share of all boxes sorted correctly:

| | single | fixed | coordinated |
|---|---|---|---|
| all 36 streams | 40.6 % | 72.2 % | 72.8 % |
| 0.05 m/s | 54.4 % | 95.0 % | 95.6 % |
| 0.1 m/s | 40.0 % | 71.1 % | 71.1 % |
| 0.15 m/s | 27.2 % | 50.6 % | 51.7 % |
| spacing 0.3 m | 35.2 % | 62.2 % | 62.6 % |
| spacing 0.6 m | 45.9 % | 82.2 % | 83.0 % |
| gray-heavy colors | 21.7 % | 68.9 % | 68.9 % |
| green-heavy colors | 52.2 % | 71.1 % | 72.8 % |
| throughput [boxes/min] | 3.3 | 5.9 | 6.0 |

Every one of the 1079 attempts ended with the box in the right region (grasp success and sorting accuracy
1.0); boxes are lost only because no robot was free in time. Planning takes 0.15 s per pick-and-place on average
(0.65 s at most). Coordinated: 8 shared-area conflicts and 22 s of waiting in total, all in the green-heavy
streams.

Three more streams (0.03 / 0.05 / 0.08 m/s, 20 boxes) contain 6 blocked pairs each, a gray box 8 cm in front of
a colored one: single 35 %, fixed 40 %, coordinated 53 % sorted. Clear-and-sort was attempted for 4 of the 18
pairs and succeeded 4 times; the other pairs were outside its limits (far half of the belt, 0.08 m/s stream).

### Prediction benchmark

```bash
pip install -e ".[eval]"
python evaluation/benchmark_prediction.py     # results/prediction.csv and evaluation/figures/prediction_error.png
```

![prediction error against horizon](evaluation/figures/prediction_error.png)

Mean position error 3 s ahead (the lead time of a pick), 500 trials: without noise both predictors are exact.
With 2 mm detection noise the window fit stays at 6.5 to 6.8 mm however long the box has been tracked; the
Kalman filter is equal after 1 s of tracking (6.9 mm) and reaches 4.1 mm after 2 s and 4.0 mm after 5 s.
With 5 mm noise: 16.3 to 16.9 mm against 17.2, 10.3 and 10.0 mm.

## Synthetic dataset for YOLO

Labels come from the simulator's ground-truth segmentation, so nothing is labelled by hand:

```bash
python scripts/generate_dataset.py --n 2000 --out data/boxes     # about 0.3 s per image
python scripts/generate_dataset.py --n 8 --out data/preview --preview   # draws the labels for a visual check
```

Classes: `0 red, 1 green, 2 blue, 3 gray`. Output is in Ultralytics layout with a `data.yaml`.

## YOLO detector

```bash
pip install -e ".[learning]"
python scripts/generate_dataset.py --n 2000 --out data/boxes
python scripts/train_yolo.py --data data/boxes/data.yaml --epochs 30     # needs a GPU; about 14 min on an RTX 3070
```

Fine-tunes the pretrained `yolo11n` on the synthetic dataset and copies the best weights to
`weights/yolo_boxes.pt`, where `YoloDetector` looks for them. The weights (5.6 MB) are not in git.

Validation (400 images, 1793 boxes): mAP50 0.995, mAP50-95 0.995, precision and recall 0.999. On 25 new random
scenes with the arms over the belt: 115 of 115 well-visible boxes found, no false positives, position error
1.7 mm mean / 3.6 mm max, 29 ms per frame on the GPU.

## SAM segmentation

`SamDetector` prompts SAM with the YOLO boxes and localizes each box from its mask. The SAM weights are
downloaded by Ultralytics into `weights/` on first use (MobileSAM 41 MB, SAM-B 375 MB).

Measured on 15 new random scenes (74 well-visible boxes, RTX 3070 Laptop GPU), all boxes found, no false positives:

| detector | position error mean / max | mask IoU mean / min | time per frame |
|---|---|---|---|
| YOLO, depth inside the bounding box | 1.7 / 3.6 mm | (no mask) | 29 ms |
| YOLO + MobileSAM (1280 px) | 1.7 / 3.8 mm | 0.975 / 0.933 | 117 ms |
| YOLO + SAM-B (1024 px) | 1.8 / 3.6 mm | 0.972 / 0.849 | 281 ms |

## Milestones

1. RAI setup and stationary KOMO pick-and-place
2. Conveyor motion, visual detection, tracking, pose prediction
3. Perception/prediction comparison, single-robot sorting, command mode,
   then Robot 2, coordination, clear-and-sort, full benchmarking
