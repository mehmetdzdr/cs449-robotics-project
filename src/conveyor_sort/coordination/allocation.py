"""Task allocation: which free robot picks which tracked box next, with a priority rule for the shared area."""
from dataclasses import dataclass, field

from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.planning.reach import can_pick
from conveyor_sort.sim.regions import DEFAULT_TARGET
from conveyor_sort.sim.world import World

LEAD_TIME = 3.0        # time from the decision to the grasp [s]
MIN_TRACK_AGE = 1.0    # a track must have been followed this long before it is picked (velocity estimate settled) [s]
MAX_TRACK_GAP = 0.2    # ... and must have been seen this recently [s]
MAX_TRACK_GAP_BOTOP = 0.7   # ... on the physics backend: BotOp renders a new camera image only every 0.35 to 0.5 s (#94)
# Part of the belt (x) where each robot picks [m]. The robots stand on opposite sides of the belt at x = 0 (#154) and
# both reach x in [-0.6, 0.6] on its centre line: Robot 1 picks upstream, Robot 2 downstream, both in the middle band.
PICK_X = {1: (-1.1, 0.15), 2: (-0.15, 1.1)}
# The fingers close along the belt: a neighbour closer than BLOCK_X in front of or behind a box, in the same lane
# (lateral offset below BLOCK_Y), is hit by them (measured: hit at 0.10 m, free at 0.12 m). Such a box is not picked.
BLOCK_X, BLOCK_Y = 0.11, 0.06
# Shared area: the space over the belt. The robots stand on opposite sides of it at x = 0 (#154): every approach from
# home swings over the middle of the belt, and the arms reach over the belt towards each other. Each arm plans with
# the other one static, so two arms there at the same time collide. Only one arm may be over the belt at a time
# (`PickPlaceTask.at_belt`: from the start of the approach until the lift); the other one may place on its own side
# meanwhile. Rules by pick position did not work: with a band around the overlap of the pick zones, or within 0.3 m
# of the other robot's zone, the hands met over the belt on the way to picks 0.31 m apart. No region is shared.

COLORED = (BoxColor.RED, BoxColor.GREEN, BoxColor.BLUE)
FIXED_ROLES = {1: COLORED, 2: (BoxColor.GRAY,)}                    # Robot 1 sorts, Robot 2 removes gray boxes


def max_track_gap(world: World) -> float:
    """How recently a track must have been seen to be picked, for this world's backend [s]."""
    return MAX_TRACK_GAP_BOTOP if world.bot is not None else MAX_TRACK_GAP


@dataclass
class CoordinationLog:
    conflicts: set[tuple[int, int]] = field(default_factory=set)   # (robot, track): box refused, shared area occupied
    waiting: set[int] = field(default_factory=set)   # robots left without a box by the latest allocation for that reason


def allocate(world: World, tracks: list[Track], first_seen: dict[int, float], blocked: dict[int, float],
             roles: dict[int, tuple[BoxColor, ...]], free_robots: list[int],
             busy_regions: set[str] = frozenset(), shared_area_busy: bool = False,
             log: CoordinationLog | None = None) -> list[Assignment]:
    """At most one Assignment per free robot; no box is given to two robots.

    A robot can take a box if the box has one of the robot's colors (`roles`), has been tracked for a while
    (`first_seen`: track id -> time), is not `blocked` (track id -> not before this time), has no other box
    right in front of or behind it (`BLOCK_X`: the fingers would hit it) and is reachable at
    the pick time, `LEAD_TIME` from now. A region holds one box: nothing is assigned to a region in
    `busy_regions` (the other robot is on its way there), and one call never sends two boxes to one region.
    Of all such (robot, box) pairs the most urgent one is served first: the box with the least time left
    before it leaves that robot's part of the belt.

    Priority over the belt (the shared area): one arm at a time (`shared_area_busy`: the other arm is still over
    the belt, see `PickPlaceTask.at_belt`); if both free robots want a box, the more urgent box wins and the other
    robot waits. Refused pairs and waiting robots are recorded in `log`.
    """
    pick_time = world.time + LEAD_TIME
    pairs = []
    for robot_id in free_robots:
        x_min, x_max = PICK_X[robot_id]
        for track in tracks:
            x = track.predict(pick_time)[0]
            if (track.color in roles[robot_id] and DEFAULT_TARGET[track.color] not in busy_regions
                    and world.time >= blocked.get(track.track_id, 0.0)
                    and world.time - first_seen[track.track_id] >= MIN_TRACK_AGE
                    and world.time - track.timestamp <= max_track_gap(world) and x_min <= x <= x_max):
                pairs.append(((x_max - x) / max(track.velocity[0], 1e-6), robot_id, track))

    assignments: dict[int, Assignment] = {}
    taken: set[int] = set()
    regions = set(busy_regions)
    refused: set[int] = set()
    for _, robot_id, track in sorted(pairs, key=lambda pair: pair[0]):
        region = DEFAULT_TARGET[track.color]
        if robot_id in assignments or track.track_id in taken or region in regions:
            continue
        position = track.predict(pick_time)
        if any(other.track_id != track.track_id and abs(other.predict(pick_time)[0] - position[0]) < BLOCK_X
               and abs(other.predict(pick_time)[1] - position[1]) < BLOCK_Y for other in tracks):
            continue
        if can_pick(world.C, robot_id, position):
            if shared_area_busy:                           # every pick uses the space over the belt
                refused.add(robot_id)
                if log:
                    log.conflicts.add((robot_id, track.track_id))
                continue
            shared_area_busy = True
            assignments[robot_id] = Assignment(robot_id, track.track_id, region, pick_time, position)
            taken.add(track.track_id)
            regions.add(region)
    if log:
        log.waiting = refused - set(assignments)
    return list(assignments.values())
