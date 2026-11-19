"""Friction grasp in the pick pipeline (#91): the box is held by the fingers, decided physically."""
from dataclasses import dataclass

import numpy as np
import pytest

from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.planning.pick_place import PickPlaceTask
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World

PICK_X = -0.3     # Robot 1 picks here [m]
SETTLE = 0.5      # boxes settle on the belt before the track is made [s]


@dataclass
class Trial:
    success: bool
    failed_stage: str | None
    lift: float           # box height gain from pick time to the end [m]
    finger_q: float       # Robot 1's finger joint at the end [m]
    box_on_belt: bool


def physical_pick(speed: float, y: float = 0.0, lead: float = 2.5, across_belt: bool = False, friction=None,
                  with_box: bool = True, dt: float = 0.05) -> Trial:
    """Robot 1 picks a box (6 cm, 0.1 kg) that reaches PICK_X at the pick time, then lifts it 15 cm."""
    world = World(ConveyorConfig(speed=speed, max_boxes=0), dt=dt, physics=True, friction=friction)
    try:
        x0 = PICK_X - speed * (SETTLE + lead)
        box = world.conveyor.spawn_box(BoxColor.RED, x=x0, y=y) if with_box else None
        world.run(SETTLE)
        position = world.C.getFrame(box).getPosition().copy() if box else np.array([x0 + speed * SETTLE, 0.2 + y, 0.65])
        track = Track(0, BoxColor.RED, position, np.array([speed, 0.0, 0.0]), world.time)
        pick_time = world.time + lead
        task = PickPlaceTask(world, Assignment(1, 0, "region_A", pick_time, track.predict(pick_time)), track,
                             across_belt=across_belt, pick_only=True)
        z_at_pick = None
        while task.advance():
            if z_at_pick is None and box and world.time >= pick_time:
                z_at_pick = world.C.getFrame(box).getPosition()[2]
            world.step()
        r = task.result
        lift = world.C.getFrame(box).getPosition()[2] - z_at_pick if box and z_at_pick is not None else 0.0
        return Trial(r.success, r.failed_stage, float(lift), float(world.C.getJointState()[world.fingers[1]]),
                     bool(box) and box in world.conveyor.on_belt())
    finally:
        world.close()


@pytest.mark.parametrize("speed, lead", [(0.0, 2.5), (0.05, 2.5), (0.1, 2.5), (0.2, 2.5), (0.3, 2.0)])
def test_box_is_picked_and_held_by_friction(speed, lead):
    trial = physical_pick(speed, lead=lead)
    print(f"speed {speed}: {trial}")
    assert trial.success and trial.failed_stage is None
    assert trial.lift == pytest.approx(0.15, abs=0.01)


@pytest.mark.parametrize("y", [-0.1, 0.1])
def test_lateral_offset(y):
    trial = physical_pick(0.1, y=y)
    print(f"y {y}: {trial}")
    assert trial.success and trial.lift == pytest.approx(0.15, abs=0.01)


def test_grasp_across_the_belt():
    trial = physical_pick(0.1, across_belt=True)
    print(f"across the belt: {trial}")
    assert trial.success and trial.lift == pytest.approx(0.15, abs=0.01)


def test_missed_grasp_is_reported():
    trial = physical_pick(0.1, with_box=False)           # the track points at a box that is not there
    print(f"no box: {trial}")
    assert not trial.success and trial.failed_stage == "grasp"


def test_slip_with_low_friction_is_reported():
    trial = physical_pick(0.1, friction=0.05)
    print(f"friction 0.05: {trial}")
    assert not trial.success and trial.failed_stage == "slip" and trial.lift < 0.05
