"""Autonomous sorting loop for one robot: detect, track, choose a box, pick and place it, repeat."""
from dataclasses import dataclass, field
from typing import Callable

from conveyor_sort.interfaces import Assignment, BoxColor, Detection, Track
from conveyor_sort.planning.pick_place import pick_and_place
from conveyor_sort.planning.reach import is_reachable
from conveyor_sort.prediction.tracker import Tracker
from conveyor_sort.sim.regions import DEFAULT_TARGET, region_contains
from conveyor_sort.sim.world import World

LEAD_TIME = 3.0        # time from the decision to the grasp [s]
MIN_TRACK_AGE = 1.0    # a track must have been followed this long before it is picked (velocity estimate settled) [s]
REACH_MARGIN = [0.1, 0.0, 0.15]   # safety margin of the pick position: downstream and up (lift height) [m]
MAX_TRACK_GAP = 0.2    # ... and must have been seen this recently [s]


@dataclass
class SortEvent:
    time: float                # simulation time when the attempt ended [s]
    color: BoxColor            # color reported by the detector
    region: str
    success: bool              # pick-and-place completed
    correct: bool              # ground truth: the box lies in the region that belongs to its true color
    failed_stage: str | None
    planning_time: float


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
                 robot_id: int, colors: set[BoxColor], attempted: set[int]) -> Track | None:
    """The most downstream box of a wanted color that the robot can still reach at the pick time."""
    pick_time = world.time + LEAD_TIME
    candidates = [
        t for t in tracks
        if t.color in colors and t.track_id not in attempted
        and world.time - first_seen[t.track_id] >= MIN_TRACK_AGE
        and world.time - t.timestamp <= MAX_TRACK_GAP
    ]
    for track in sorted(candidates, key=lambda t: -t.predict(pick_time)[0]):
        position = track.predict(pick_time)
        # at the edge of the workspace a box can be reached but not lifted: also require a point
        # further downstream and at lift height to be reachable
        if is_reachable(world.C, robot_id, position) and is_reachable(world.C, robot_id, position + REACH_MARGIN):
            return track
    return None


def run_single_robot(world: World, detect: Callable[[], list[Detection]], duration: float, robot_id: int = 1,
                     colors: tuple[BoxColor, ...] = (BoxColor.RED, BoxColor.GREEN, BoxColor.BLUE),
                     on_step: Callable[[World], None] | None = None) -> SortingStats:
    """Sort boxes of the given colors into their default regions for `duration` seconds of simulation time.

    `detect()` returns the detections of the current camera frame (ground truth or a vision detector).
    """
    tracker = Tracker()
    stats = SortingStats()
    first_seen: dict[int, float] = {}
    attempted: set[int] = set()            # every track is tried once: a failed plan must not be retried forever
    tracks: list[Track] = []
    last_placed: dict[str, str] = {}       # region -> box lying in it
    end_time = world.time + duration
    missed_before = world.conveyor.missed

    def perceive(world: World) -> None:          # runs after every simulation step, also while the arm moves
        tracks[:] = tracker.update(detect(), world.time)
        for track in tracks:
            first_seen.setdefault(track.track_id, world.time)
        if on_step:
            on_step(world)

    while world.time < end_time:
        track = choose_track(world, tracks, first_seen, robot_id, set(colors), attempted)
        if track is None:
            world.step()
            perceive(world)
            continue
        attempted.add(track.track_id)
        region = DEFAULT_TARGET[track.color]
        if region in last_placed:              # regions are emptied (box collected) before the next box arrives
            world.conveyor.remove(last_placed.pop(region))
        pick_time = world.time + LEAD_TIME
        result = pick_and_place(world, Assignment(robot_id, track.track_id, region, pick_time, track.predict(pick_time)),
                                track, perceive)
        correct = (result.success and DEFAULT_TARGET[world.conveyor.colors[result.box]] == region
                   and region_contains(world.C, region, world.C.getFrame(result.box).getPosition()))
        if result.success:
            last_placed[region] = result.box
        stats.events.append(SortEvent(world.time, track.color, region, result.success, correct,
                                      result.failed_stage, result.planning_time))
    stats.missed = world.conveyor.missed - missed_before
    return stats
