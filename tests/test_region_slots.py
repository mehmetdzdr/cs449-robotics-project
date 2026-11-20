"""Regions that hold several boxes (#93): one box per slot, none knocked out, a full region is reported."""
import numpy as np

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


def test_region_fills_slot_by_slot_and_reports_full():
    world = World(ConveyorConfig(speed=0.0, max_boxes=0), physics=True)
    try:
        placed, results = [], []
        for _ in range(len(SLOTS) + 1):
            box = world.conveyor.spawn_box(BoxColor.RED, x=-0.3, y=0.0)
            world.run(0.5)
            track = Track(0, BoxColor.RED, world.C.getFrame(box).getPosition().copy(), np.zeros(3), world.time)
            pick_time = world.time + 2.5
            task = PickPlaceTask(world, Assignment(1, 0, "region_A", pick_time, track.predict(pick_time)), track)
            while task.advance():
                world.step()
            results.append(task.result)
            print(f"box {len(results)}: {task.result.failed_stage}, error {task.result.place_error}, tilt {task.result.tilt}")
            if task.result.success:
                placed.append(box)
                world.run(0.5)
                for slot, b in enumerate(placed):                       # nothing knocked out of its slot
                    frame = world.C.getFrame(b)
                    assert np.all(np.abs(frame.getPosition()[:2] - slot_position(world.C, "region_A", slot))
                                  < SLOT_TOLERANCE) and tilt(frame) < MAX_TILT
            else:
                world.conveyor.remove(box)
    finally:
        world.close()
    assert all(r.success and r.place_error < 0.02 for r in results[:-1])   # every slot filled
    assert results[-1].failed_stage == "full" and len(placed) == len(SLOTS)  # then reported, not overfilled
