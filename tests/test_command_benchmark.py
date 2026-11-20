import pytest

from conveyor_sort.command.benchmark import random_commands, run_commands
from conveyor_sort.command.parser import parse_command
from conveyor_sort.interfaces import Command


def test_random_commands_cover_colors_and_regions_and_parse_back():
    commands = random_commands(200, seed=1)
    assert random_commands(200, seed=1) == commands and random_commands(200, seed=2) != commands
    assert len({color for _, color, _ in commands}) == 4 and len({region for _, _, region in commands}) == 4
    assert len({text for text, _, _ in commands}) > 100                    # many phrasings
    for text, color, region in commands:
        assert parse_command(text) == Command(color, region), text


def test_random_commands_are_carried_out_correctly():
    records = run_commands(random_commands(6, seed=3))
    assert len(records) == 6
    for record in records:
        assert record["parsed"] and record["success"] and record["correct_color"] and record["correct_region"], record
        assert record["robot"] in (1, 2) and record["failed_stage"] is None and 5.0 < record["time"] < 90.0
    rate = sum(r["correct_color"] and r["correct_region"] for r in records) / len(records)
    print(f"success rate {rate:.0%}, robots {[r['robot'] for r in records]}")


@pytest.mark.slow
def test_commands_on_physics_with_blocked_boxes():
    """#109: the command benchmark on the physics backend, every box blocked by a gray box in front of it."""
    records = run_commands(random_commands(2, seed=0), speed=0.1, physics=True, blocked_pairs=1.0)
    print([(r["command"], r["failed_stage"], r["time"]) for r in records])
    assert len(records) == 2 and all(r["physics"] and r["blocked_pairs"] == 1.0 for r in records)
    assert all(r["parsed"] for r in records) and sum(r["correct_color"] and r["correct_region"] for r in records) >= 1
