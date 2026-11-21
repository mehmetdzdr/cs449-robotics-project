"""Scenario matrix (#106): speed x lateral x neighbourhood on the opposite-robots layout (#154), with the expected
outcome of every cell (`scenario_matrix.expected`). The cells and the physics runs are in
evaluation/scenario_matrix.py, which also writes the results table."""
import sys
from pathlib import Path

import numpy as np
import pytest

from conveyor_sort.coordination.allocation import allocate
from conveyor_sort.coordination.clear_and_sort import plan_clearing_job
from conveyor_sort.interfaces import Track
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.scenarios import make_matrix_scenario
from conveyor_sort.sim.world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
from scenario_matrix import ROLES, cells, decision_x, expected, run_cell  # noqa: E402


@pytest.fixture(scope="module")
def decision_world():
    world = World(ConveyorConfig(max_boxes=0))
    world.time = 5.0
    return world


def test_the_matrix_covers_the_issue():
    """Speeds {0.05, 0.1, 0.2, 0.3} x lateral {centre, near edge, far edge} x neighbourhood {free, front, back,
    front+back, sides, surrounded}, plus front_side (both grasp axes blocked by one front and one side box); side
    neighbours do not fit beside a target at the belt edge."""
    layouts = {(layout, lateral) for layout, lateral, _ in cells()}
    assert len(cells()) == 4 * len(layouts) == 68
    assert ("left_right", "near_edge") not in layouts and ("front_side", "far_edge") in layouts


@pytest.mark.parametrize("layout, lateral, speed", cells(), ids=str)
def test_planning_decision(decision_world, layout, lateral, speed):
    """Every cell, planning only: what allocate and plan_clearing_job decide when the target is grasped at x = -0.2."""
    scenario = make_matrix_scenario(layout, seed=0, lateral=lateral, x=decision_x(speed))
    boxes = [(scenario.target_color, scenario.target)] + scenario.neighbours
    tracks = [Track(i, color, np.array([*position, 0.65]), np.array([speed, 0.0, 0.0]), decision_world.time)
              for i, (color, position) in enumerate(boxes)]
    seen = {t.track_id: 0.0 for t in tracks}
    assignments = allocate(decision_world, tracks, seen, {}, ROLES, [1, 2])
    decision = plan_clearing_job(decision_world, tracks, seen, {}, ROLES, 0)
    outcome = expected(layout, speed)
    if outcome == "missed":                                    # too fast (allocation.MAX_PICK_SPEED): left alone
        assert assignments == [] and decision.job is None and decision.infeasible == []
    elif layout in ("free", "left_right"):                     # grasped along the belt by allocate
        assert [(a.robot_id, a.track_id) for a in assignments] == [(1, 0)]
        assert decision.job is None and decision.infeasible == []
    elif outcome.startswith("sorted"):                         # front / back: grasped across the belt, no removal
        assert assignments == [] and len(decision.job.steps) == 1
        assert (decision.job.steps[0].robot_id, decision.job.steps[0].track_id) == (1, 0)
        assert decision.job.steps[0].across_belt
    elif outcome == "cleared":                                 # Robot 2 bins one gray box, then Robot 1 sorts
        assert assignments == [] and len(decision.job.steps) == 2
        assert (decision.job.steps[0].robot_id, decision.job.steps[0].region) == (2, "bin_gray")
        assert (decision.job.steps[1].robot_id, decision.job.steps[1].track_id) == (1, 0)
    else:
        assert assignments == [] and decision.job is None and decision.infeasible == [0] and not decision.later


@pytest.mark.slow
@pytest.mark.parametrize("cell", cells(), ids=str)
def test_on_physics(cell):
    """Every cell on BotOp physics: the expected outcome, no arm-arm contact, no box left in a gripper."""
    record = run_cell(cell)
    print(record)
    assert record["arm_contacts"] == 0 and not record["box_held"]
    assert record["ok"], f"expected {record['expected']}, observed {record['observed']} ({record['failed_stages']})"
