"""Autonomous sorting loop: detect, track, and for every robot choose a box of its colors, pick and place it, repeat."""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from conveyor_sort.interfaces import Assignment, BoxColor, Detection, Track
from conveyor_sort.planning.pick_place import PickPlaceTask
from conveyor_sort.planning.reach import is_reachable
from conveyor_sort.prediction.tracker import Tracker
from conveyor_sort.sim.regions import DEFAULT_TARGET, region_contains
from conveyor_sort.sim.world import World

LEAD_TIME = 3.0        # time from the decision to the grasp [s]
MIN_TRACK_AGE = 1.0    # a track must have been followed this long before it is picked (velocity estimate settled) [s]
REACH_MARGIN = [0.1, 0.0, 0.15]   # safety margin of the pick position: downstream and up (lift height) [m]
MAX_TRACK_GAP = 0.2    # ... and must have been seen this recently [s]
PICK_X_MIN = {2: 0.4}  # no coordination yet: Robot 2 picks only downstream of Robot 1's reach (x <= 0.2), with clearance [m]
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

    @property
    def sorted_correctly(self) -> int:
        return sum(e.correct for e in self.events)

    @property
    def failed(self) -> int:
        return sum(not e.success for e in self.events)


def choose_track(world: World, tracks: list[Track], first_seen: dict[int, float],
                 robot_id: int, colors: set[BoxColor], blocked: dict[int, float]) -> Track | None:
    """The most downstream box of a wanted color that the robot can still reach at the pick time.

    `blocked` maps a track id to the time before which the track must not be chosen."""
    pick_time = world.time + LEAD_TIME
    candidates = [
        t for t in tracks
        if t.color in colors and world.time >= blocked.get(t.track_id, 0.0)
        and world.time - first_seen[t.track_id] >= MIN_TRACK_AGE
        and world.time - t.timestamp <= MAX_TRACK_GAP
    ]
    for track in sorted(candidates, key=lambda t: -t.predict(pick_time)[0]):
        position = track.predict(pick_time)
        if position[0] < PICK_X_MIN.get(robot_id, -np.inf):
            continue
        # at the edge of the workspace a box can be reached but not lifted: also require a point
        # further downstream and at lift height to be reachable
        if is_reachable(world.C, robot_id, position) and is_reachable(world.C, robot_id, position + REACH_MARGIN):
            return track
    return None


COLORED = (BoxColor.RED, BoxColor.GREEN, BoxColor.BLUE)
FIXED_ROLES = {1: COLORED, 2: (BoxColor.GRAY,)}    # Robot 1 sorts, Robot 2 removes gray boxes


def run_robots(world: World, detect: Callable[[], list[Detection]], duration: float,
               roles: dict[int, tuple[BoxColor, ...]] = FIXED_ROLES,
               on_step: Callable[[World], None] | None = None) -> SortingStats:
    """Every robot in `roles` (robot_id -> colors) sorts boxes of its colors into their default regions for
    `duration` seconds of simulation time. The robots work at the same time and do not coordinate.

    `detect()` returns the detections of the current camera frame (ground truth or a vision detector).
    """
    tracker = Tracker()
    stats = SortingStats()
    first_seen: dict[int, float] = {}
    blocked: dict[int, float] = {}         # track id -> not before this time; forever once the arm has moved for it
    tracks: list[Track] = []
    last_placed: dict[str, str] = {}       # region -> box lying in it
    tasks: dict[int, tuple[PickPlaceTask, Track]] = {}
    end_time = world.time + duration
    missed_before = world.conveyor.missed

    def advance(robot_id: int) -> None:
        """Move the robot by one step; when it is free, start on the next box."""
        while True:
            if robot_id not in tasks:
                track = choose_track(world, tracks, first_seen, robot_id, set(roles[robot_id]), blocked)
                if track is None:
                    return
                blocked[track.track_id] = float("inf")
                region = DEFAULT_TARGET[track.color]
                if region in last_placed:      # regions are emptied (box collected) before the next box arrives
                    world.conveyor.remove(last_placed.pop(region))
                pick_time = world.time + LEAD_TIME
                assignment = Assignment(robot_id, track.track_id, region, pick_time, track.predict(pick_time))
                tasks[robot_id] = PickPlaceTask(world, assignment, track), track
            task, track = tasks[robot_id]
            if task.advance():
                return
            del tasks[robot_id]
            result, region = task.result, task.assignment.target_region
            if result.failed_stage == "pick":      # no plan from here (e.g. box just entering the workspace): no
                blocked[track.track_id] = world.time + RETRY_DELAY   # time was used, try again a little later
                continue
            correct = (result.success and DEFAULT_TARGET[world.conveyor.colors[result.box]] == region
                       and region_contains(world.C, region, world.C.getFrame(result.box).getPosition()))
            if result.success:
                last_placed[region] = result.box
            stats.events.append(SortEvent(world.time, track.color, region, result.success, correct,
                                          result.failed_stage, result.planning_time, robot_id))

    while world.time < end_time:
        for robot_id in roles:
            advance(robot_id)
        world.step()
        tracks[:] = tracker.update(detect(), world.time)
        for track in tracks:
            first_seen.setdefault(track.track_id, world.time)
        if on_step:
            on_step(world)
    stats.missed = world.conveyor.missed - missed_before
    return stats


def run_single_robot(world: World, detect: Callable[[], list[Detection]], duration: float, robot_id: int = 1,
                     colors: tuple[BoxColor, ...] = COLORED,
                     on_step: Callable[[World], None] | None = None) -> SortingStats:
    """Single-robot baseline: only `robot_id` works."""
    return run_robots(world, detect, duration, {robot_id: colors}, on_step)
