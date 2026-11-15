"""Autonomous sorting loop: detect, track, allocate boxes to free robots, pick and place them, repeat."""
from dataclasses import dataclass, field
from typing import Callable

from conveyor_sort.coordination.allocation import (COLORED, FIXED_ROLES, LEAD_TIME, CoordinationLog, allocate,
                                                    uses_shared_area)
from conveyor_sort.coordination.clear_and_sort import plan_clear_and_sort
from conveyor_sort.interfaces import Assignment, BoxColor, Detection, Track
from conveyor_sort.planning.pick_place import PickPlaceTask
from conveyor_sort.prediction.tracker import Tracker
from conveyor_sort.sim.regions import DEFAULT_TARGET, region_contains
from conveyor_sort.sim.world import World

RETRY_DELAY = 0.5      # a box whose pick could not be planned is tried again after this time [s]


@dataclass
class SortEvent:
    time: float                # simulation time when the attempt ended [s]
    color: BoxColor            # color reported by the detector
    region: str
    success: bool              # pick-and-place completed
    correct: bool              # ground truth: the box lies in the region that belongs to its true color
    failed_stage: str | None
    planning_time: float
    robot_id: int = 1


@dataclass
class SortingStats:
    events: list[SortEvent] = field(default_factory=list)
    missed: int = 0            # boxes that left the belt end (any color)
    conflicts: int = 0         # (robot, box) pairs refused at least once because the other robot held the shared area
    waiting_time: float = 0.0  # robot time spent idle only because the shared area was occupied [s]
    clear_and_sort: list[bool] = field(default_factory=list)   # one entry per attempt: wanted box sorted after the gray one was removed

    @property
    def sorted_correctly(self) -> int:
        return sum(e.correct for e in self.events)

    @property
    def failed(self) -> int:
        return sum(not e.success for e in self.events)


def run_robots(world: World, detect: Callable[[], list[Detection]], duration: float,
               roles: dict[int, tuple[BoxColor, ...]] = FIXED_ROLES,
               on_step: Callable[[World], None] | None = None, clear_and_sort: bool = False) -> SortingStats:
    """Every robot in `roles` (robot_id -> colors) sorts boxes of its colors into their default regions for
    `duration` seconds of simulation time. The robots work at the same time; `allocate` decides who takes
    which box (with `SHARED_ROLES` green boxes go to whichever robot is free). A box with a neighbour right in
    front of or behind it is left alone, except with `clear_and_sort`: if the neighbour is gray, Robot 2 removes
    it first and Robot 1 then sorts the box (needs both robots).

    `detect()` returns the detections of the current camera frame (ground truth or a vision detector).
    """
    tracker = Tracker()
    stats = SortingStats()
    first_seen: dict[int, float] = {}
    blocked: dict[int, float] = {}         # track id -> not before this time; forever once the arm has moved for it
    tracks: list[Track] = []
    last_placed: dict[str, str] = {}       # region -> box lying in it
    tasks: dict[int, tuple[PickPlaceTask, Track]] = {}
    scheduled: dict[int, tuple[float, Assignment]] = {}   # clear-and-sort: robot -> (time to start moving, assignment)
    pair: dict = {}                        # the running clear-and-sort: track ids "gray" and "wanted", "gray_ok"
    log = CoordinationLog()
    end_time = world.time + duration
    missed_before = world.conveyor.missed

    def finish(robot_id: int) -> None:
        task, track = tasks.pop(robot_id)
        result, region = task.result, task.assignment.target_region
        if result.failed_stage == "pick" and track.track_id not in pair.values():   # no plan from here (e.g. box just
            blocked[track.track_id] = world.time + RETRY_DELAY   # entering the workspace): no time was used, try again later
            return
        correct = (result.success and DEFAULT_TARGET[world.conveyor.colors[result.box]] == region
                   and region_contains(world.C, region, world.C.getFrame(result.box).getPosition()))
        if result.success:
            last_placed[region] = result.box
        stats.events.append(SortEvent(world.time, track.color, region, result.success, correct,
                                      result.failed_stage, result.planning_time, robot_id))
        if track.track_id == pair.get("gray"):
            pair["gray_ok"] = result.success
            if not result.success and scheduled.pop(1, None):    # blocker still there: Robot 1 does not start
                stats.clear_and_sort.append(False)
                pair.clear()
        elif track.track_id == pair.get("wanted"):
            stats.clear_and_sort.append(correct and pair["gray_ok"])
            pair.clear()

    def start(assignment: Assignment, across_belt: bool = False) -> None:
        track = by_id[assignment.track_id]
        if assignment.target_region in last_placed:   # regions are emptied (box collected) before the next box arrives
            world.conveyor.remove(last_placed.pop(assignment.target_region))
        tasks[assignment.robot_id] = PickPlaceTask(world, assignment, track, across_belt), track

    while world.time < end_time:
        moved: set[int] = set()            # robots that have made their move for this simulation step
        log.waiting = set()
        while True:
            by_id = {track.track_id: track for track in tracks}
            for robot_id, (start_time, assignment) in list(scheduled.items()):
                if world.time >= start_time - 1e-9:
                    del scheduled[robot_id]
                    if assignment.track_id in by_id:
                        start(assignment, across_belt=robot_id == 2)   # Robot 2: the wanted box is in the way along the belt
                    else:                                              # track lost while waiting
                        stats.clear_and_sort.append(False)
                        scheduled.clear()
                        pair.clear()
            free = [robot_id for robot_id in roles if robot_id not in tasks and robot_id not in scheduled]
            if clear_and_sort and not pair and set(free) >= {1, 2}:
                plan = plan_clear_and_sort(world, tracks, first_seen, blocked, roles[1])
                if plan:
                    pair.update(gray=plan[0].track_id, wanted=plan[1].track_id, gray_ok=False)
                    for assignment in plan:
                        blocked[assignment.track_id] = float("inf")
                        scheduled[assignment.robot_id] = assignment.pick_time - LEAD_TIME, assignment
                    free = []
            busy_regions = {task.assignment.target_region for task, _ in tasks.values()}
            shared_busy = any(uses_shared_area(task.assignment.pick_position[0], task.assignment.target_region)
                              for task, _ in tasks.values())
            for assignment in (allocate(world, tracks, first_seen, blocked, roles, free, busy_regions, shared_busy, log)
                               if free else []):
                blocked[assignment.track_id] = float("inf")
                start(assignment)
            pending = [robot_id for robot_id in tasks if robot_id not in moved]
            if not pending:
                break
            for robot_id in pending:       # a task that ends (or cannot be planned) uses no time: allocate again
                if tasks[robot_id][0].advance():
                    moved.add(robot_id)
                else:
                    finish(robot_id)
        stats.waiting_time += world.dt * len(log.waiting - set(tasks))
        world.step()
        tracks[:] = tracker.update(detect(), world.time)
        for track in tracks:
            first_seen.setdefault(track.track_id, world.time)
        if on_step:
            on_step(world)
    stats.missed = world.conveyor.missed - missed_before
    stats.conflicts = len(log.conflicts)
    return stats


def run_single_robot(world: World, detect: Callable[[], list[Detection]], duration: float, robot_id: int = 1,
                     colors: tuple[BoxColor, ...] = COLORED,
                     on_step: Callable[[World], None] | None = None) -> SortingStats:
    """Single-robot baseline: only `robot_id` works."""
    return run_robots(world, detect, duration, {robot_id: colors}, on_step)
