"""Command-mode correctness: random commands on a running belt, checked against ground truth."""
import numpy as np

from conveyor_sort.command.mode import CommandMode
from conveyor_sort.interfaces import BoxColor
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.regions import region_contains
from conveyor_sort.sim.world import World

# {color} and the destination phrase are filled in; the phrasings exercise the parser too
TEMPLATES = ["pick a {color} box and place it {to}", "put a {color} box {to}", "Move the {color} box {to}.",
             "take a {color} one {to}", "{to} place a {color} box"]
DESTINATIONS = {"region_A": ["at A", "to region A", "into region a"], "region_B": ["at B", "to region B", "in region b"],
                "region_C": ["at C", "to region C", "on region c"], "bin_gray": ["into the bin", "to the gray bin", "in the bin"]}


def random_commands(n: int, seed: int = 0) -> list[tuple[str, BoxColor, str]]:
    """`n` commands as (text, color, region), color and region uniformly random, random phrasing."""
    rng = np.random.default_rng(seed)
    commands = []
    for _ in range(n):
        color = list(BoxColor)[rng.integers(len(BoxColor))]
        region = list(DESTINATIONS)[rng.integers(len(DESTINATIONS))]
        phrase = DESTINATIONS[region][rng.integers(len(DESTINATIONS[region]))]
        commands.append((TEMPLATES[rng.integers(len(TEMPLATES))].format(color=color.value, to=phrase), color, region))
    return commands


def run_commands(commands: list[tuple[str, BoxColor, str]], speed: float = 0.1, spacing: float = 0.4, seed: int = 0,
                 timeout: float = 90.0, physics: bool = False, blocked_pairs: float = 0.0) -> list[dict]:
    """Execute the commands one after the other on one belt (ground-truth detections); one record per command.

    `correct_color` and `correct_region` are checked on the simulator state, not on what the pipeline reports.
    `physics`: BotOp physics backend (friction grasp, physical place). `blocked_pairs`: share of spawns that are a
    gray box with a colored box 8 cm behind it (1.0: every box on the belt is blocked along the belt, #109).
    """
    world = World(ConveyorConfig(speed=speed, spacing=spacing, seed=seed, blocked_pairs=blocked_pairs), physics=physics)
    try:
        mode = CommandMode(world, GroundTruthDetector(world).detect)
        records = []
        for text, color, region in commands:
            result = mode.execute(text, timeout)
            placed = result.success and result.box is not None
            records.append({
                "command": text, "color": color.value, "region": region,
                "parsed": result.command.color == color and result.command.target_region == region,
                "success": result.success,
                "correct_color": placed and world.conveyor.colors[result.box] == color,
                "correct_region": placed and region_contains(world.C, region, world.C.getFrame(result.box).getPosition()),
                "robot": result.robot_id, "failed_stage": result.failed_stage, "time": round(result.time, 2),
                "speed": speed, "physics": physics, "blocked_pairs": blocked_pairs, "seed": seed,
            })
        return records
    finally:
        world.close()
