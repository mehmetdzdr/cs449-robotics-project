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
python scripts/run_command.py "pick a blue box and place it at B" --view   # command mode
python scripts/run_sorting.py --view    # Robot 1 sorts colored boxes (--detector hsv for vision, --robots 2 adds Robot 2 for gray boxes)
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

## Experiments

```bash
python evaluation/run_experiment.py evaluation/configs/example.json --out results/runs.csv
```

The JSON config sets belt speed, box spacing, number of boxes, color distribution, seed and detector. Each run
appends one row to the CSV: grasp success rate, sorting accuracy, missed boxes, planning time, completion time
and throughput (boxes per minute).

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

## Milestones

1. RAI setup and stationary KOMO pick-and-place
2. Conveyor motion, visual detection, tracking, pose prediction
3. Perception/prediction comparison, single-robot sorting, command mode,
   then Robot 2, coordination, clear-and-sort, full benchmarking
