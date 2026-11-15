import pytest

from conveyor_sort.command.parser import parse_command
from conveyor_sort.interfaces import BoxColor, Command
from conveyor_sort.sim.regions import REGIONS


@pytest.mark.parametrize("text, color, region", [
    ("pick a blue box and place it at B", BoxColor.BLUE, "region_B"),
    ("Pick a red box and place it at A.", BoxColor.RED, "region_A"),
    ("put a GREEN box into region C", BoxColor.GREEN, "region_C"),
    ("move the gray box to the bin", BoxColor.GRAY, "bin_gray"),
    ("pick a grey box and place it in the gray bin", BoxColor.GRAY, "bin_gray"),
    ("pick a red box and place it in the gray bin", BoxColor.RED, "bin_gray"),
    ("take a blue one to region a", BoxColor.BLUE, "region_A"),
    ("at C place a green box", BoxColor.GREEN, "region_C"),
])
def test_valid_commands(text, color, region):
    assert parse_command(text) == Command(color, region)
    assert region in REGIONS


@pytest.mark.parametrize("text, message", [
    ("pick a box and place it at B", "no box color"),
    ("pick a yellow box and place it at B", "no box color"),
    ("pick a red or blue box and place it at B", "more than one box color"),
    ("pick a blue box", "no destination"),
    ("pick a blue box and place it at D", "no destination"),
    ("pick a blue box and place it at A or at B", "more than one destination"),
    ("", "no box color"),
])
def test_invalid_commands_are_rejected_with_a_clear_message(text, message):
    with pytest.raises(ValueError, match=message):
        parse_command(text)
