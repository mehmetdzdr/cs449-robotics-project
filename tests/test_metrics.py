import csv
import json
import math

import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.metrics import ExperimentConfig, append_csv, load_config, run_experiment, summarize
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import SortEvent, SortingStats


def test_conveyor_stops_spawning_at_max_boxes():
    world = World(ConveyorConfig(speed=0.5, spacing=0.3, max_boxes=4))
    world.run(3.0)
    assert world.conveyor.spawned == 4 and len(world.conveyor.on_belt()) == 4
    world.run(5.0)
    assert world.conveyor.spawned == 4 and world.conveyor.missed == 4


def test_summarize():
    events = [
        SortEvent(10.0, BoxColor.RED, "region_A", True, True, None, 0.2),
        SortEvent(20.0, BoxColor.BLUE, "region_C", True, False, None, 0.4),      # placed, but the box was not blue
        SortEvent(25.0, BoxColor.GREEN, "region_B", False, False, "grasp", 0.1),
        SortEvent(40.0, BoxColor.GREEN, "region_B", False, False, "place", 0.3),
    ]
    m = summarize(SortingStats(events, missed=2), boxes=6, duration=120.0)
    assert (m["boxes"], m["attempts"], m["sorted_correctly"], m["missed"]) == (6, 4, 1, 2)
    assert m["grasp_success_rate"] == 0.75
    assert m["sorting_accuracy"] == 0.5
    assert m["planning_time_mean"] == pytest.approx(0.25) and m["planning_time_max"] == 0.4
    assert m["completion_time"] == 40.0
    assert m["throughput"] == 0.5
    assert math.isnan(summarize(SortingStats(), boxes=0, duration=10.0)["grasp_success_rate"])


def test_config_file_sets_the_experiment_and_the_record_is_logged(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"name": "sparse", "speed": 0.1, "spacing": 1.2, "boxes": 3, "seed": 0,
         "color_weights": {"red": 1, "green": 1, "blue": 1}}))
    config = load_config(tmp_path / "config.json")
    assert config == ExperimentConfig("sparse", 0.1, 1.2, 3, {"red": 1, "green": 1, "blue": 1}, 0)

    record = run_experiment(config)
    assert record["duration"] == pytest.approx((2 * 1.2 + 2.2) / 0.1 + 12.0)
    assert (record["boxes"], record["attempts"], record["sorted_correctly"], record["missed"]) == (3, 3, 3, 0)
    assert record["grasp_success_rate"] == 1.0 and record["sorting_accuracy"] == 1.0
    assert 0 < record["planning_time_mean"] <= record["planning_time_max"] < 5.0
    assert 30.0 < record["completion_time"] < record["duration"]
    assert record["throughput"] == pytest.approx(3 / record["duration"] * 60)

    append_csv(tmp_path / "results" / "runs.csv", record)
    append_csv(tmp_path / "results" / "runs.csv", record)
    rows = list(csv.DictReader((tmp_path / "results" / "runs.csv").open()))
    assert len(rows) == 2 and rows[0]["name"] == "sparse" and float(rows[1]["sorting_accuracy"]) == 1.0
    assert json.loads(rows[0]["color_weights"]) == {"red": 1, "green": 1, "blue": 1}
    json.dumps(record)                                       # the record is also valid JSON


def test_unknown_config_key_or_detector_is_an_error(tmp_path):
    (tmp_path / "config.json").write_text('{"sped": 0.1}')
    with pytest.raises(TypeError):
        load_config(tmp_path / "config.json")
    with pytest.raises(ValueError, match="unknown detector"):
        run_experiment(ExperimentConfig(detector="yolo9000"))


def test_blocked_pairs_in_the_box_stream():
    world = World(ConveyorConfig(speed=0.5, spacing=0.4, blocked_pairs=1.0, max_boxes=6, seed=3))
    world.run(4.0)
    names = sorted(world.conveyor.on_belt(), key=lambda n: -world.C.getFrame(n).getPosition()[0])   # first box first
    assert world.conveyor.pairs == 3 and len(names) == 6
    for gray, colored in zip(names[::2], names[1::2]):
        assert world.conveyor.colors[gray] == BoxColor.GRAY and world.conveyor.colors[colored] != BoxColor.GRAY
        offset = world.C.getFrame(gray).getPosition() - world.C.getFrame(colored).getPosition()
        assert offset == pytest.approx([0.08, 0, 0], abs=0.03)             # gray right in front, same lane


def test_three_manipulation_configurations_on_the_same_stream():
    records = {mode: run_experiment(ExperimentConfig("modes", 0.1, 0.45, 10, seed=0, mode=mode))
               for mode in ("single", "fixed", "coordinated")}
    assert records["single"]["sorted_correctly"] == 5 and records["single"]["missed"] == 5      # the gray boxes leave
    assert records["fixed"]["sorted_correctly"] == 9 and records["fixed"]["missed"] == 1     # robots facing each other (#154)
    assert records["coordinated"]["sorted_correctly"] >= 8
    for record in records.values():
        assert record["grasp_success_rate"] == 1.0 and record["sorting_accuracy"] == 1.0
        assert record["sorted_fraction"] == record["sorted_correctly"] / 10
        assert record["conflicts"] >= 0 and record["waiting_time"] >= 0.0
        assert record["clear_and_sort_attempts"] == 0 and math.isnan(record["clear_and_sort_success_rate"])
    with pytest.raises(ValueError, match="unknown mode"):
        run_experiment(ExperimentConfig(mode="swarm"))


def test_clear_and_sort_is_logged_for_a_stream_with_blocked_pairs():
    record = run_experiment(ExperimentConfig("pairs", 0.05, 0.6, 10, seed=0, mode="coordinated", blocked_pairs=0.5))
    assert record["blocked_pairs_spawned"] == 3
    assert record["clear_and_sort_attempts"] == 3 and record["clear_and_sort_success_rate"] == 1.0   # every pair (#154)
    fixed = run_experiment(ExperimentConfig("pairs", 0.05, 0.6, 10, seed=0, mode="fixed", blocked_pairs=0.5))
    assert fixed["clear_and_sort_attempts"] == 0 and fixed["sorted_correctly"] < record["sorted_correctly"]


@pytest.mark.slow
def test_physics_run_logs_grasp_failures_slips_drops_and_tracking_error():
    """#107: the same experiment on the BotOp physics backend, with the new metrics."""
    record = run_experiment(ExperimentConfig("physics", 0.1, 0.6, 4, seed=0, mode="fixed", physics=True))
    print({k: record[k] for k in ("sorted_correctly", "missed", "grasp_failures", "slips", "drops",
                                  "tracking_error_mean", "arm_contacts")})
    assert record["physics"] and record["boxes"] == 4 and record["sorted_correctly"] >= 3
    failures = record["grasp_failures"] + record["slips"] + record["drops"] + record["other_failures"]
    assert failures <= record["attempts"]
    assert 0.0 <= record["tracking_error_mean"] < 0.005 and record["arm_contacts"] == 0   # ground-truth detections
    assert sum(json.loads(record["dropped_on"]).values()) == record["slips"] + record["drops"]
