"""Scripted belt layouts (#98): reproducible, on the belt, no interpenetration, still on a stopped belt."""
import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.sim.conveyor import BOX_SIZE, ConveyorConfig
from conveyor_sort.sim.scenarios import LATERAL, NEIGHBOURHOODS, SPACING, make_scenario, scenarios, spawn_scenario
from conveyor_sort.sim.world import World


def offsets(scenario) -> list[tuple[float, float]]:
    return [tuple(np.round(p - scenario.target, 3)) for _, p in scenario.neighbours]


def test_lateral_positions_and_spacings():
    assert [make_scenario("free", lateral=l).target[1] for l in LATERAL] == pytest.approx([0.2, 0.08, 0.32])
    for spacing, distance in [("touching", 0.061), ("gap_3cm", 0.09), ("centre_11cm", 0.11), ("centre_30cm", 0.30)]:
        assert offsets(make_scenario("front", spacing=spacing)) == [(distance, 0.0)]


def test_neighbourhoods():
    d = SPACING["touching"]
    assert offsets(make_scenario("free")) == []
    assert offsets(make_scenario("back")) == [(-round(d, 3), 0.0)]
    assert sorted(offsets(make_scenario("front_back"))) == [(-round(d, 3), 0.0), (round(d, 3), 0.0)]
    assert sorted(offsets(make_scenario("left_right"))) == [(0.0, -round(d, 3)), (0.0, round(d, 3))]
    assert len(make_scenario("surrounded").neighbours) == 4 and len(make_scenario("cluster_3x3").neighbours) == 8
    scenario = make_scenario("cluster_3x3")
    assert all(color == BoxColor.GRAY for color, _ in scenario.neighbours) and scenario.target_color != BoxColor.GRAY


def test_reproducible_and_seed_picks_the_target_color():
    a, b = make_scenario("surrounded", seed=3), make_scenario("surrounded", seed=3)
    assert a.target_color == b.target_color and np.array_equal(a.target, b.target)
    assert [(c, tuple(p)) for c, p in a.neighbours] == [(c, tuple(p)) for c, p in b.neighbours]
    assert len({make_scenario("free", seed=s).target_color for s in range(30)}) == 3
    assert make_scenario("front", neighbour_color=BoxColor.RED).neighbours[0][0] == BoxColor.RED


def test_layouts_that_do_not_fit_on_the_belt_are_refused():
    with pytest.raises(ValueError, match="does not fit"):
        make_scenario("left_right", lateral="far_edge")
    with pytest.raises(ValueError, match="does not fit"):
        make_scenario("cluster_3x3", spacing="centre_30cm")
    with pytest.raises(ValueError, match="unknown"):
        make_scenario("pyramid")
    every = scenarios()
    assert len(every) == 48 and {s.name for s in every} == set(NEIGHBOURHOODS)
    assert {s.lateral for s in every if s.name == "front_back"} == set(LATERAL)


def settles_in_place(scenario) -> tuple[float, float, float]:
    """Spawn on a stopped physics belt, wait 1 s: (deepest box-box penetration, largest drift [m], largest tilt [deg])."""
    world = World(ConveyorConfig(speed=0.0, max_boxes=0), physics=True)
    try:
        target, neighbours = spawn_scenario(world, scenario)
        boxes = [target] + neighbours
        world.run(0.3)
        start = {b: world.C.getFrame(b).getPosition().copy() for b in boxes}
        world.run(1.0)
        world.C.computeCollisions()
        depth = min([d for a, b, d in world.C.getCollisions(0.0) if a in boxes and b in boxes] + [0.0])
        drift = max(np.linalg.norm(world.C.getFrame(b).getPosition() - start[b]) for b in boxes)
        tilt = max(np.degrees(np.arccos(np.clip(world.C.getFrame(b).getRotationMatrix()[2, 2], -1, 1))) for b in boxes)
        assert set(world.conveyor.on_belt()) == set(boxes)
        return -depth, drift, tilt
    finally:
        world.close()


def check_still(scenario):
    depth, drift, tilt = settles_in_place(scenario)
    print(f"{scenario.name}/{scenario.lateral}/{scenario.spacing}: penetration {depth * 1000:.2f} mm, "
          f"drift {drift * 1000:.2f} mm, tilt {tilt:.2f} deg")
    assert depth < 0.001 and drift < 0.002 and tilt < 1.0


@pytest.mark.parametrize("name, lateral, spacing", [("cluster_3x3", "centre", "touching"),
                                                    ("surrounded", "centre", "gap_3cm"),
                                                    ("front_back", "near_edge", "touching")])
def test_dense_layouts_spawn_without_interpenetration_and_stay_still(name, lateral, spacing):
    check_still(make_scenario(name, lateral=lateral, spacing=spacing))


@pytest.mark.slow
@pytest.mark.parametrize("scenario", scenarios(), ids=lambda s: f"{s.name}-{s.lateral}-{s.spacing}")
def test_every_layout_spawns_without_interpenetration_and_stays_still(scenario):
    check_still(scenario)
