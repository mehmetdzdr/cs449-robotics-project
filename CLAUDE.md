# CLAUDE.md

CS449/549 term project (Bilkent): vision-based multi-robot sorting on a conveyor belt in RAI (`robotic`), planning with KOMO.
Proposal summary, pipeline and module owners are in `README.md`.

## Commands

```bash
pip install -e ".[dev]"                      # core deps; ".[learning]" adds YOLO / SAM
python scripts/check_env.py                  # must print ENVIRONMENT OK
pytest -q                                    # headless Linux: xvfb-run -a pytest -q
python scripts/run_conveyor.py --view        # conveyor only (or --video out.mp4 headless)
python scripts/run_sorting.py --view          # Robot 1 sorts colored boxes (--robots 2: Robot 2 removes gray; --detector gt|hsv, --video out.mp4)
python scripts/run_command.py "pick a blue box and place it at B" --view   # command mode (no command: type them in)
python scripts/generate_dataset.py --n 2000 --out data/boxes [--preview]
python evaluation/run_experiment.py evaluation/configs/example.json   # one run from a JSON config, metrics appended to results/runs.csv
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
| `sim/conveyor.py` | `Conveyor`: `spawn_box`, `step`, `on_belt`, `attach`, `remove`, `missed`, `spawned`; `ConveyorConfig` (`max_boxes` ends the stream); `belt_geometry` |
| `sim/regions.py` | `REGIONS`, `DEFAULT_TARGET` (red A, green B, blue C, gray bin), `region_contains` |
| `sim/world.py` | `World`: scene + conveyor + clock (`step`, `run`) |
| `sim/ground_truth.py` | `GroundTruthDetector(world).detect()` |
| `perception/camera.py` | `Camera.capture()` returns `Observation` (rgb, depth, GT segmentation, intrinsics, pose) |
| `perception/localization.py` | `mask_to_world(mask, obs)`: mask + depth to box centre in world frame |
| `perception/hsv.py` | `HsvDetector(belt).detect(obs)`: HSV thresholds in the belt ROI, blobs, height check |
| `perception/yolo.py` | `YoloDetector(belt, weights).detect(obs)`: fine-tuned YOLO11n (`weights/yolo_boxes.pt`, not in git); `boxes_to_detections`: bounding boxes + depth to `Detection`s. `scripts/train_yolo.py` trains it |
| `perception/dataset.py` | Synthetic YOLO dataset, labels from GT segmentation |
| `prediction/tracker.py` | `Tracker(predictor="cv" or "kalman").update(detections, time)`: association + constant-velocity window fit or Kalman filter |
| `prediction/benchmark.py` | `prediction_errors(predictor, noise, horizons, frames)`: future-position errors on a synthetic box; `evaluation/benchmark_prediction.py` writes `results/prediction.csv` and `evaluation/figures/prediction_error.png` (needs `.[eval]`) |
| `planning/reach.py`, `planning/pick.py` | `is_reachable`, `plan_pick` (KOMO reach to a stationary box), `plan_moving_pick` (moving box, from a `Track`), `plan_lift`, `plan_place`, `plan_home` |
| `command/parser.py` | `parse_command(text)`: text to `Command` (color + region), `ValueError` with a user message if invalid |
| `command/selection.py` | `select(world, command, tracks, first_seen, robots, blocked)`: `Assignment` for the box of the commanded color closest to a robot that reaches the target region (`regions.REGION_ROBOTS`), or None |
| `command/mode.py` | `CommandMode(world, detect).execute(command or text, timeout)`: waits for a feasible box, picks and places it, returns `CommandResult`; several commands in a row on the running belt |
| `planning/execute.py` | `execute_path(world, path, duration)`, `follow_path` (generator, one robot's joints only), `grasp(world, robot_id, box=None)` (returns box name or None), `release(world, box, region)` (kinematic) |
| `metrics.py` | `ExperimentConfig` / `load_config(json)`, `run_experiment(config)` returns one flat record (config + metrics), `summarize(stats, boxes, duration)`, `append_csv` |
| `coordination/allocation.py` | `allocate(world, tracks, first_seen, blocked, roles, free_robots, busy_regions, shared_area_busy, log)`: at most one `Assignment` per free robot, most urgent (robot, box) pair first, one robot at a time in the shared area (`uses_shared_area`, `CoordinationLog`); `FIXED_ROLES`, `SHARED_ROLES` (both robots sort green), `PICK_X`, `LEAD_TIME` |
| `sorting.py` | `run_robots(world, detect, duration, roles=FIXED_ROLES)`: tracker + `allocate` + one `PickPlaceTask` per robot, both arms move at once, returns `SortingStats`; `run_single_robot` |
| `planning/pick_place.py` | `PickPlaceTask(world, assignment, track)`: non-blocking, `advance()` before every `world.step()`, moves only its own arm; `pick_and_place(...)` is the blocking form, returns `PickPlaceResult` |

## Facts worth knowing

- Grasping is kinematic by team decision (no physics engine): the box is attached to the gripper.
- Stationary pick-and-place sequence: `plan_pick`, `grasp`, `plan_lift`, `plan_place`, `release`, `plan_lift` (retreat up, otherwise the fingers hit the box), `plan_home`; see `tests/test_place.py`.
- `plan_moving_pick` path must be executed in exactly `pick_time - now` seconds; the planner treats the box as static at its current position for collision checks.
- Never `delFrame` while planning is in use: RAI's collision engine keeps stale geometry (wrong KOMO results or a crash). `Conveyor.remove` parks boxes below the floor instead. Only `perception/dataset.py` deletes frames (no planning there).
- A box can be reachable but not liftable at the edge of the workspace; `allocation.allocate` therefore requires a point 10 cm downstream and 15 cm up to be reachable too.
- Each arm plans with the other arm static at its current pose, so two arms working close together collide. Priority rule: the shared area (belt x in (-0.3, 0.4) and region B) is used by one robot at a time, for its whole pick-and-place; the other robot may only take boxes outside of it. `SortingStats.conflicts` / `waiting_time` log it. Robot 2 picks only at x >= 0.4 (`allocation.PICK_X`): nearer to its reach limit (-0.2) the lift fails and the arm is left holding the box.
- A pick plan can fail where IK says reachable (box just entering a robot's workspace): the sorting loop retries that track 0.5 s later instead of giving up.
- A region holds one box: the sorting loop removes the previous box before the next one is placed there.
- Rendering costs about 0.15 s per frame, so vision-in-the-loop runs are slow (36 s of simulation took about 2 min).
- YOLO: a bounding box can contain part of an arm above the box; `boxes_to_detections` localizes from the pixels at box-top height only. The model test is skipped when the weights file is missing (CI). Python 3.14 starts data-loader workers by re-importing the script: training code needs `if __name__ == "__main__"`.
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

Done and merged (PRs #44 to #70, 105 tests, CI green):
M0 #2 #3 · M1 #4 to #9 · M2 #10 to #14 (complete) · M3 #15 #16 #17 #18 · M4 #23 #24 #25 (complete) · M5 #26 #27 · M6 #30 · M7 #34

Measured: localization error 1.6 mm mean / 2.8 mm max (GT masks); HSV detector 2.0 mm mean / 3.5 mm max, no misses or false positives on 124 well-visible boxes; prediction error at 2 s with 2 mm synthetic noise: 4.8 mm with the window fit, 3.1 mm with Kalman after 40 frames (equal up to 20 frames); KOMO pick planning about 0.05 s; a full pick-and-place needs about 0.25 s planning and 10 to 11 s of simulation time, so one robot sorts about one box per 10 s. Two robots with fixed roles (a box every 6 s at 0.1 m/s, 60 s): 7 of 7 sorted, no collisions; Robot 1 alone misses the 3 gray boxes.

Open:
- #1 stays open until every member has run `check_env.py` on their own machine.
- Merged branches still exist on GitHub (delete manually or enable auto-delete).
- `robotic` has Linux wheels only: on Windows use WSL.

## Next steps, in order

Critical path: #31 -> #32 -> #33 -> #37 -> #41 -> #43.

1. #31 task allocation (replaces the fixed `PICK_X_MIN` split), #28 command-mode box and robot selection, #36 prediction benchmark.
2. #19 YOLO fine-tune: needs a GPU machine with internet for pretrained weights; dataset generator is ready. Then #20 SAM, #22 detector factory (`metrics.run_experiment` builds its detector inline until then), #35.
3. #32 shared-workspace priority, #29 command mode end-to-end, then #33 clear-and-sort, #38.
4. #37 manipulation benchmark (`metrics.run_experiment` only runs the single-robot loop so far), #39 to #43.
