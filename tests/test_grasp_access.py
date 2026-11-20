"""Grasp accessibility (#99): free grasp axes and blockers for every scripted layout of #98."""
import numpy as np
import pytest

from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.planning.access import grasp_access
from conveyor_sort.planning.pick_place import PickPlaceTask
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.scenarios import make_scenario, scenarios, spawn_scenario
from conveyor_sort.sim.world import World

BOTH = ["along", "across"]
EXPECTED_FREE = {   # (neighbourhood, spacing) -> free axes; 11 cm and 30 cm leave both axes free
    "touching": {"free": BOTH, "front": ["across"], "back": ["across"], "front_back": ["across"],
                 "left_right": ["along"], "surrounded": [], "cluster_3x3": []},
    # a 3 cm gap blocks only the grasp along the belt (see planning/access.py)
    "gap_3cm": {"free": BOTH, "front": ["across"], "back": ["across"], "front_back": ["across"],
                "left_right": BOTH, "surrounded": ["across"], "cluster_3x3": ["across"]},
}


def access(scenario):
    return grasp_access(scenario.target, scenario.neighbours)


@pytest.mark.parametrize("scenario", scenarios(), ids=lambda s: f"{s.name}-{s.lateral}-{s.spacing}")
def test_free_axes_for_every_layout(scenario):
    expected = EXPECTED_FREE.get(scenario.spacing, {}).get(scenario.name, BOTH)
    assert access(scenario).free == expected


def test_blockers_are_listed_per_axis():
    scenario = make_scenario("surrounded")          # neighbours: front, back, left, right
    result = access(scenario)
    assert result.free == [] and result.blockers == {"along": [0, 1], "across": [2, 3]}
    cluster = make_scenario("cluster_3x3")          # the diagonal boxes block neither axis
    blocking = access(cluster).blockers["along"] + access(cluster).blockers["across"]
    diagonal = [i for i, (_, p) in enumerate(cluster.neighbours) if np.all(np.abs(p - cluster.target) > 0.01)]
    assert len(diagonal) == 4 and not set(diagonal) & set(blocking) and len(blocking) == 4


def test_a_blocker_of_another_color_is_reported():
    scenario = make_scenario("front", neighbour_color=BoxColor.RED)
    result = access(scenario)
    assert result.free == ["across"] and [scenario.neighbours[i][0] for i in result.blockers["along"]] == [BoxColor.RED]


def physical_grasp(layout: str, spacing: str, across: bool, speed: float = 0.1) -> str | None:
    """Robot 1 grasps the target of the layout on the moving belt (physics); returns the failed stage or None."""
    world = World(ConveyorConfig(speed=speed, max_boxes=0), physics=True)
    try:
        target, _ = spawn_scenario(world, make_scenario(layout, spacing=spacing, x=-0.3 - speed * 3.0))
        world.run(0.5)
        track = Track(0, BoxColor.RED, world.C.getFrame(target).getPosition().copy(), np.array([speed, 0, 0]),
                      world.time)
        pick_time = world.time + 2.5
        task = PickPlaceTask(world, Assignment(1, 0, "region_A", pick_time, track.predict(pick_time)), track,
                             across_belt=across, pick_only=True)
        while task.advance():
            world.step()
        return task.result.failed_stage
    finally:
        world.close()


@pytest.mark.slow
@pytest.mark.parametrize("layout, spacing, axis", [("front_back", "gap_3cm", "along"),
                                                   ("left_right", "gap_3cm", "across"),
                                                   ("left_right", "touching", "across")])
def test_blocking_limits_on_physics(layout, spacing, axis):
    """The limits of planning/access.py hold for a physical grasp at 0.1 m/s."""
    free = axis in access(make_scenario(layout, spacing=spacing)).free
    stage = physical_grasp(layout, spacing, across=axis == "across")
    print(f"{layout} {spacing}, grasp {axis}: {stage or 'held'} (grasp_access: {'free' if free else 'blocked'})")
    assert (stage is None) == free
