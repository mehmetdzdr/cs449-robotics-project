"""Two arms at once under physics (#95): the shared-area rule keeps the arms apart, with real tracking errors."""
import pytest

import conveyor_sort.sorting as sorting
from conveyor_sort.coordination.allocation import SHARED_ROLES, uses_shared_area
from conveyor_sort.interfaces import BoxColor
from conveyor_sort.planning.pick_place import PickPlaceTask
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.ground_truth import GroundTruthDetector
from conveyor_sort.sim.world import World


def run(world, duration, monkeypatch, **kwargs):
    """run_robots with SHARED_ROLES; also returns the time intervals of the tasks that used the shared area."""
    shared: list[list[float]] = []      # [start, end] per task that uses the shared area

    class RecordedTask(PickPlaceTask):
        def __init__(self, world, assignment, track, across_belt=False):
            super().__init__(world, assignment, track, across_belt)
            self.interval = None
            if uses_shared_area(assignment.pick_position[0], assignment.target_region):
                self.interval = [world.time, float("inf")]
                shared.append(self.interval)
            self.world = world

        def advance(self):
            running = super().advance()
            if not running and self.interval:
                self.interval[1] = self.world.time
            return running

    monkeypatch.setattr(sorting, "PickPlaceTask", RecordedTask)
    stats = sorting.run_robots(world, GroundTruthDetector(world).detect, duration, SHARED_ROLES, **kwargs)
    overlaps = sum(a[0] < b[1] and b[0] < a[1] for i, a in enumerate(shared) for b in shared[i + 1:])
    return stats, len(shared), overlaps


@pytest.mark.parametrize("speed", [0.1, 0.2])
def test_coordinated_run_has_no_arm_contact(speed, monkeypatch):
    world = World(ConveyorConfig(speed=speed, spacing=0.6, seed=0), physics=True)
    try:
        stats, shared, overlaps = run(world, 60.0, monkeypatch, clear_and_sort=True)
    finally:
        world.close()
    by_robot = {r: sum(e.correct for e in stats.events if e.robot_id == r) for r in (1, 2)}
    print(f"{speed} m/s: {stats.sorted_correctly} of {len(stats.events)} sorted (robot 1: {by_robot[1]}, robot 2: "
          f"{by_robot[2]}), failed stages {[e.failed_stage for e in stats.events if not e.success]}, closest arm "
          f"distance {stats.min_arm_distance * 100:.1f} cm, {shared} shared-area tasks, conflicts {stats.conflicts}, "
          f"waiting {stats.waiting_time:.1f} s")
    assert stats.arm_contacts == [] and stats.min_arm_distance > 0.0
    assert overlaps == 0
    assert by_robot[1] >= 2 and by_robot[2] >= 1             # both arms really worked


def test_arm_in_the_shared_area_holds_the_other_robot(monkeypatch):
    """Robot 1 picks a red box inside the shared area while Robot 2's green box goes to region B (shared)."""
    world = World(ConveyorConfig(speed=0.03, max_boxes=0), physics=True)   # slow: both boxes stay reachable
    try:
        world.conveyor.spawn_box(BoxColor.RED, x=-0.35, y=0.0)    # picked in the shared area (x > -0.3)
        world.conveyor.spawn_box(BoxColor.GREEN, x=0.3, y=0.0)    # Robot 2 picks it (x >= 0.4), places it in B
        stats, shared, overlaps = run(world, 30.0, monkeypatch)
    finally:
        world.close()
    print(f"conflicts {stats.conflicts}, waiting {stats.waiting_time:.1f} s, "
          f"{[(e.robot_id, e.region, e.correct) for e in stats.events]}")
    assert sorted((e.robot_id, e.region) for e in stats.events if e.correct) == [(1, "region_A"), (2, "region_B")]
    assert stats.conflicts >= 1 and stats.waiting_time > 1.0   # one robot waited for the other ...
    assert shared == 2 and overlaps == 0                        # ... until the first had left the shared area
    assert stats.arm_contacts == []
