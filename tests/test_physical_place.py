"""Physical place and release (#92): the box is lowered into the region, let go and must rest upright inside it."""
import numpy as np
import pytest

from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.planning.pick_place import MAX_TILT, PickPlaceTask
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.regions import region_contains
from conveyor_sort.sim.world import World

PICK_X = {1: -0.3, 2: 0.6}   # each robot picks a box resting here on the stopped belt [m]


def physical_pick_place(robot_id: int, region: str, drop: float = 0.0, speed: float = 0.0, settle: float = 1.0):
    """One pick and place of a box (6 cm, 0.1 kg) that reaches PICK_X at pick time; returns (result, box tilt,
    inside region, box parent) after the arm is home and the box had `settle` more seconds."""
    world = World(ConveyorConfig(speed=speed, max_boxes=0), physics=True)
    try:
        box = world.conveyor.spawn_box(BoxColor.RED, x=PICK_X[robot_id] - speed * 3.0, y=0.0)
        world.run(0.5)
        track = Track(0, BoxColor.RED, world.C.getFrame(box).getPosition().copy(), np.array([speed, 0, 0]),
                      world.time)
        pick_time = world.time + 2.5
        task = PickPlaceTask(world, Assignment(robot_id, 0, region, pick_time, track.predict(pick_time)), track,
                             drop=drop)
        while task.advance():
            world.step()
        world.run(settle)
        frame = world.C.getFrame(box)
        tilt = float(np.degrees(np.arccos(np.clip(frame.getRotationMatrix()[2, 2], -1, 1))))
        return task.result, tilt, region_contains(world.C, region, frame.getPosition()), frame.getParent()
    finally:
        world.close()


@pytest.mark.parametrize("robot_id, region", [(1, "region_A"), (1, "region_B"), (1, "region_C"),
                                              (2, "region_B"), (2, "bin_gray")])
def test_box_rests_upright_in_the_region(robot_id, region):
    result, tilt, inside, parent = physical_pick_place(robot_id, region)
    print(f"robot {robot_id} -> {region}: {result}, tilt {tilt:.1f} deg 1 s later")
    assert result.success and result.failed_stage is None
    assert result.place_error < 0.02 and result.tilt < MAX_TILT
    assert inside and tilt < MAX_TILT                       # the box stays put after the arm has left
    assert parent is None                                    # no kinematic release: the box is a free body


def test_box_from_the_moving_belt():
    result, tilt, inside, _ = physical_pick_place(1, "region_A", speed=0.2)
    print(f"0.2 m/s: {result}, tilt {tilt:.1f} deg")
    assert result.success and inside and tilt < MAX_TILT and result.place_error < 0.02


def test_box_dropped_from_2_cm_is_accepted():
    result, tilt, inside, _ = physical_pick_place(1, "region_A", drop=0.02)
    print(f"dropped from 2 cm: {result}, tilt {tilt:.1f} deg")
    assert result.success and inside and tilt < MAX_TILT
