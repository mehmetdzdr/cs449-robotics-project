"""PickPlaceTask decisions that must not move an arm (physics backend; the kinematic task was removed in #113). The
full physical pick-and-place is tested in test_physical_place.py, recovery in test_recovery.py."""
import numpy as np
import pytest

from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.planning.pick_place import PickPlaceTask
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.world import World


@pytest.fixture
def physics_world():
    worlds = []

    def make(**config) -> World:
        worlds.append(World(ConveyorConfig(max_boxes=0, **config), physics=True))
        return worlds[-1]
    yield make
    for world in worlds:
        world.close()


def assignment_for(track, robot_id, region, pick_time):
    return Assignment(robot_id, track.track_id, region, pick_time, track.predict(pick_time))


def arm(world, robot_id):
    return world.arms.reference(robot_id, world.bot.get_t())


def test_the_task_needs_the_physics_backend():
    world = World(ConveyorConfig(max_boxes=0))
    track = Track(0, BoxColor.RED, np.array([-0.3, 0.2, 0.65]), np.zeros(3), 0.0)
    with pytest.raises(ValueError, match="physics"):
        PickPlaceTask(world, assignment_for(track, 1, "region_A", 3.0), track)


@pytest.mark.slow
def test_unreachable_assignment_fails_at_pick_without_moving(physics_world):
    world = physics_world(speed=0.1)
    box = world.conveyor.spawn_box(BoxColor.GRAY, x=-1.0, y=0.0)
    world.run(0.5)
    track = Track(0, BoxColor.GRAY, world.C.getFrame(box).getPosition(), np.array([0.1, 0, 0]), world.time)
    q, time = arm(world, 2), world.time
    task = PickPlaceTask(world, assignment_for(track, 2, "bin_gray", world.time + 2.0), track)   # at x = -0.75
    assert not task.advance()
    assert task.result.failed_stage == "pick" and task.result.box is None
    assert arm(world, 2) == pytest.approx(q) and world.time == time and box in world.conveyor.on_belt()


@pytest.mark.slow
def test_pick_is_not_started_when_the_lift_from_the_grasp_pose_cannot_be_planned(physics_world):
    """Far side of the belt at the upstream limit of Robot 1: the grasp pose is reachable, but the arm cannot lift
    from it (#154: x = -0.5); 15 cm further on the pick starts."""
    world = physics_world(speed=0.05)
    box = world.conveyor.spawn_box(BoxColor.BLUE, x=-0.625, y=0.07)
    world.run(0.5)

    def task_now():                    # grasp 2.5 s from now: at x = -0.5, after 3 s more at -0.35
        track = Track(0, BoxColor.BLUE, world.C.getFrame(box).getPosition(), np.array([0.05, 0, 0]), world.time)
        return PickPlaceTask(world, assignment_for(track, 1, "region_C", world.time + 2.5), track)

    q = arm(world, 1)
    task = task_now()
    assert not task.advance() and task.result.failed_stage == "pick" and arm(world, 1) == pytest.approx(q)
    world.run(3.0)
    assert task_now().advance()                                                   # the arm starts moving
