"""Minimal set of blockers to remove (#100), on the scripted layouts of #98."""
import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.planning.access import grasp_access
from conveyor_sort.planning.clearing import plan_clearing
from conveyor_sort.sim.scenarios import make_scenario, scenarios


def plan(scenario, **kwargs):
    return plan_clearing(scenario.target, scenario.neighbours, **kwargs)


def test_surrounded_on_four_sides_needs_two_removals_on_one_axis():
    scenario = make_scenario("surrounded")              # neighbours 0 front, 1 back, 2 left, 3 right
    result = plan(scenario)
    assert len(result.removals) == 2
    assert {r.box for r in result.removals} in ({0, 1}, {2, 3})
    axis_freed = "along" if {r.box for r in result.removals} == {0, 1} else "across"
    assert result.target_axis == axis_freed


def test_front_and_back_only_needs_no_removal():
    result = plan(make_scenario("front_back"))
    assert result.removals == [] and result.target_axis == "across"


def test_touching_3x3_cluster_is_infeasible():
    """Every box of a touching 3 x 3 cluster is blocked in both axes (edge boxes by the target and two corner boxes,
    corner boxes by their two edge neighbours): nothing can be taken away first."""
    assert plan(make_scenario("cluster_3x3")) is None


def test_3x3_cluster_with_gaps():
    assert plan(make_scenario("cluster_3x3", spacing="gap_3cm")).removals == []   # 3 cm gaps: across is free
    result = plan(make_scenario("cluster_3x3", spacing="centre_11cm"))
    assert result.removals == [] and result.target_axis == "along"


def test_unreachable_or_expensive_blockers_are_avoided():
    scenario = make_scenario("surrounded")
    # front unreachable: the plan frees the other axis
    assert {r.box for r in plan(scenario, reachable=lambda i: i != 0).removals} == {2, 3}
    # left and right expensive (e.g. colored boxes that must be sorted instead of binned): front and back go
    assert {r.box for r in plan(scenario, cost=lambda i: 5.0 if i in (2, 3) else 1.0).removals} == {0, 1}
    assert plan(scenario, reachable=lambda i: i not in (0, 2)) is None


def test_removal_order_keeps_every_step_graspable():
    """The box in front of the target is blocked along the belt by the target and across it by two boxes beside
    it: those two must go first. The target's own side neighbours are unreachable, so the along axis is freed."""
    t = np.array([0.0, 0.2])
    gray = BoxColor.GRAY
    boxes = [(gray, t + [0, 0.085]), (gray, t + [0, -0.085]),         # 0, 1: left and right of the target
             (gray, t + [0.061, 0]), (gray, t + [-0.061, 0]),         # 2, 3: front and back
             (gray, t + [0.111, 0.075]), (gray, t + [0.111, -0.075])]  # 4, 5: beside the front box
    order = [r.box for r in plan_clearing(t, boxes, reachable=lambda i: i not in (0, 1)).removals]
    assert sorted(order) == [2, 3, 4, 5] and order.index(2) > max(order.index(4), order.index(5))
    assert sorted(r.box for r in plan_clearing(t, boxes).removals) == [0, 1]    # reachable: 2 removals suffice


@pytest.mark.parametrize("scenario", scenarios(), ids=lambda s: f"{s.name}-{s.lateral}-{s.spacing}")
def test_every_plan_is_valid(scenario):
    """For every layout of #98: each removal had a free axis when it was made, and the target has one at the end."""
    result = plan(scenario)
    if result is None:
        assert scenario.name == "cluster_3x3" and scenario.spacing == "touching"
        return
    remaining = list(range(len(scenario.neighbours)))
    for removal in result.removals:
        remaining.remove(removal.box)
        others = [scenario.neighbours[j] for j in remaining] + [(BoxColor.GRAY, scenario.target)]
        assert removal.axis in grasp_access(scenario.neighbours[removal.box][1], others).free
    final = grasp_access(scenario.target, [scenario.neighbours[j] for j in remaining]).free
    assert result.target_axis in final
