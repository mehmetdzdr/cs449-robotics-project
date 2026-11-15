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
