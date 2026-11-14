# CLAUDE.md

CS449/549 term project (Bilkent): vision-based multi-robot sorting on a conveyor belt in RAI (`robotic`), planning with KOMO.
Proposal summary, pipeline and module owners are in `README.md`.

## Commands

```bash
pip install -e ".[dev]"                      # core deps; ".[learning]" adds YOLO / SAM
python scripts/check_env.py                  # must print ENVIRONMENT OK
pytest -q                                    # headless Linux: xvfb-run -a pytest -q
python scripts/run_conveyor.py --view        # conveyor only (or --video out.mp4 headless)
python scripts/generate_dataset.py --n 2000 --out data/boxes [--preview]
```

Headless Linux needs `libglu1-mesa`, `liblapack3`, `xvfb`; every command that renders must run under `xvfb-run -a`.

## Workflow

- One small GitHub issue = one branch = one PR with `Closes #n`, squash-merged into `main`. Issues are numbered `[milestone.step]` and list their dependencies.
- Every PR adds tests; CI (`.github/workflows/ci.yml`) runs `check_env.py` and `pytest` under xvfb.
- PR descriptions state measured numbers and any deviation from the issue text.
- Minimal code, no speculative features. Modules talk only through `src/conveyor_sort/interfaces.py` (`Detection`, `Track`, `Command`, `Assignment`).

## Code map

| Path | Content |
|---|---|
| `assets/scene.g` | Our additions (top camera, belt, regions) on top of RAI's `scenarios/pandasTable.g` |
| `sim/scene.py` | `load_scene()`, `ROBOT_PREFIX` (1 = `l_`, 2 = `r_`), `HOME_Q` |
| `sim/conveyor.py` | `Conveyor`: `spawn_box`, `step`, `on_belt`, `attach`, `missed`; `ConveyorConfig`; `belt_geometry` |
| `sim/regions.py` | `REGIONS`, `DEFAULT_TARGET` (red A, green B, blue C, gray bin), `region_contains` |
| `sim/world.py` | `World`: scene + conveyor + clock (`step`, `run`) |
| `sim/ground_truth.py` | `GroundTruthDetector(world).detect()` |
| `perception/camera.py` | `Camera.capture()` returns `Observation` (rgb, depth, GT segmentation, intrinsics, pose) |
| `perception/localization.py` | `mask_to_world(mask, obs)`: mask + depth to box centre in world frame |
| `perception/hsv.py` | `HsvDetector(belt).detect(obs)`: HSV thresholds in the belt ROI, blobs, height check |
| `perception/dataset.py` | Synthetic YOLO dataset, labels from GT segmentation |
| `prediction/tracker.py` | `Tracker.update(detections, time)`: association + constant-velocity fit |
| `planning/reach.py`, `planning/pick.py` | `is_reachable`, `plan_pick` (KOMO reach to a stationary box), `plan_moving_pick` (moving box, from a `Track`), `plan_lift`, `plan_place`, `plan_home` |
| `planning/execute.py` | `execute_path(world, path, duration)`, `grasp(world, robot_id, box=None)` (returns box name or None), `release(world, box, region)` (kinematic) |
| `planning/pick_place.py` | `pick_and_place(world, assignment, track, on_step)`: plans and executes one `Assignment`, returns `PickPlaceResult` |

## Facts worth knowing

- Grasping is kinematic by team decision (no physics engine): the box is attached to the gripper.
- Stationary pick-and-place sequence: `plan_pick`, `grasp`, `plan_lift`, `plan_place`, `release`, `plan_lift` (retreat up, otherwise the fingers hit the box), `plan_home`; see `tests/test_place.py`.
- `plan_moving_pick` path must be executed in exactly `pick_time - now` seconds; the planner treats the box as static at its current position for collision checks.
- HSV detector: gray blobs on the robot arms are rejected by a height check from depth; touching boxes of the same color merge into one blob (not handled).
- `World.step()` spawns a box on the very first step even with a huge `spacing`.
- Belt: along +x, x in [-1.1, 1.1], centre y = 0.2, surface z = 0.62. Boxes: 6 cm cubes, kinematic motion (no physics).
- Reach on the belt centre line: Robot 1 x in [-1.0, 0.2], Robot 2 x in [-0.2, 1.0]. Regions A, C: Robot 1 only; B: both; bin_gray: Robot 2 only.
- Top camera: 1280 x 720, 3.4 m high, looks straight down; a box is about 30 x 30 px, one pixel about 2.2 mm.
- A box is "on the belt" while its parent frame is `belt`. Grasp / place = `Conveyor.attach(box, parent)` (`Config.attach` does not re-parent).
- A second `.g` file cannot use frames of the first as parents: root frames in `scene.g` use absolute `X:` poses.
- `ry.CameraView` renders stale colors after frames are added/removed. Always use `perception.camera.Camera`, which rebuilds the view.
- Thin `ssBox` shapes fail to load; use `box` for flat markers.
- GitHub from Claude sessions: REST only (`gh api repos/...`), no GraphQL; deleting remote branches is not permitted.

## Status (last updated 2026-11-15)

Done and merged (PRs #44 to #65, 74 tests, CI green):
M0 #2 #3 · M1 #4 to #9 · M2 #10 to #14 (complete) · M3 #15 #16 #17 #18 · M4 #23 #24

Measured: localization error 1.6 mm mean / 2.8 mm max (GT masks); HSV detector 2.0 mm mean / 3.5 mm max, no misses or false positives on 124 well-visible boxes; prediction error 4.8 mm at 2 s with 2 mm synthetic noise; KOMO pick planning about 0.05 s; a full pick-and-place needs about 0.25 s planning and 10 to 11 s of simulation time.

Open:
- #1 stays open until every member has run `check_env.py` on their own machine.
- 16 merged branches still exist on GitHub (delete manually or enable auto-delete).

## Next steps, in order

1. #26 single-robot autonomous sorting loop with the ground-truth detector (needs #22 detector factory first, or use `GroundTruthDetector` directly). `pick_and_place` blocks until the arm is home: pass `on_step` to keep the tracker updated meanwhile. For two arms moving at once (#30) it has to become non-blocking.
2. #19 YOLO fine-tune: must run on a GPU machine with internet for pretrained weights; dataset generator is ready. Then #20 SAM, #22 detector factory.
3. #25 Kalman, then M5 onwards.
