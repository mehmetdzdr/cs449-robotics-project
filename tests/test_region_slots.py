"""Regions that hold several boxes (#93): one box per slot, none knocked out, a full region is reported."""
import numpy as np
import pytest

from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.planning.pick_place import MAX_TILT, PickPlaceTask
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.regions import SLOT_TOLERANCE, SLOTS, free_slot, slot_position
from conveyor_sort.sim.world import World


def test_free_slot_skips_occupied_slots():
    world = World(ConveyorConfig(speed=0.0, max_boxes=0))
    boxes = [world.conveyor.spawn_box(BoxColor.RED) for _ in SLOTS]
    assert free_slot(world.C, "region_A", boxes) == 0                  # all boxes are on the belt
    for slot, box in enumerate(boxes):
        world.C.getFrame(box).setPosition([*(slot_position(world.C, "region_A", slot) + 0.01), 0.632])
        assert free_slot(world.C, "region_A", boxes) == (slot + 1 if slot + 1 < len(SLOTS) else None)


def tilt(frame) -> float:
    return float(np.degrees(np.arccos(np.clip(frame.getRotationMatrix()[2, 2], -1, 1))))


@pytest.mark.slow
@pytest.mark.parametrize("robot_id, region, pick_x", [(1, "region_A", -0.3), (1, "region_B", -0.3), (2, "bin_gray", 0.6)])
def test_region_fills_slot_by_slot_and_reports_full(robot_id, region, pick_x):
    """At least 5 boxes (the issue): sometimes the last free slot cannot be planned (region B, bin), then the box is
    let go of over the belt; measured 6 of 6 in region A, 5 or 6 in region B, 5 in the bin."""
    world = World(ConveyorConfig(speed=0.0, max_boxes=0), physics=True)
    try:
        placed, results = {}, []                                    # box -> its position after the place
        for _ in range(len(SLOTS) + 1):
            box = world.conveyor.spawn_box(BoxColor.RED, x=pick_x, y=0.0)
            world.run(0.5)
            track = Track(0, BoxColor.RED, world.C.getFrame(box).getPosition().copy(), np.zeros(3), world.time)
            pick_time = world.time + 2.5
            task = PickPlaceTask(world, Assignment(robot_id, 0, region, pick_time, track.predict(pick_time)), track)
            while task.advance():
                world.step()
            results.append(task.result)
            print(f"box {len(results)}: {task.result.failed_stage}, error {task.result.place_error}, tilt {task.result.tilt}")
            if task.result.success:
                placed[box] = world.C.getFrame(box).getPosition().copy()
            else:
                world.conveyor.remove(box)
            world.run(0.5)
            for b, position in placed.items():                        # nothing knocked out of its slot
                frame = world.C.getFrame(b)
                moved = np.abs(frame.getPosition()[:2] - position[:2])        # neighbours may be nudged (up to 1.8 cm
                assert np.all(moved < SLOT_TOLERANCE) and tilt(frame) < MAX_TILT   # measured) but stay in their slot
    finally:
        world.close()
    moved = max(np.linalg.norm(world.C.getFrame(b).getPosition()[:2] - p[:2]) for b, p in placed.items())
    print(f"{region}: {len(placed)} placed, largest nudge {moved * 1000:.1f} mm")
    assert len(placed) >= 5 and all(r.place_error < 0.02 for r in results if r.success)
    assert len({tuple(np.round(p[:2], 2)) for p in placed.values()}) == len(placed)  # one box per slot
    if len(placed) == len(SLOTS):
        assert results[-1].failed_stage == "full"                                  # then reported, not overfilled
