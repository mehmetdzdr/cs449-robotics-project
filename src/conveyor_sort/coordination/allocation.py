"""Task allocation: which free robot picks which tracked box next, with a priority rule for the shared area."""
from dataclasses import dataclass, field

from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.planning.reach import can_pick
from conveyor_sort.sim.regions import DEFAULT_TARGET
from conveyor_sort.sim.world import World

LEAD_TIME = 3.0        # time from the decision to the grasp [s]
MIN_TRACK_AGE = 1.0    # a track must have been followed this long before it is picked (velocity estimate settled) [s]
MAX_TRACK_GAP = 0.2    # ... and must have been seen this recently [s]
PICK_X = {1: (-1.1, 0.2), 2: (0.4, 1.1)}   # part of the belt (x) where each robot picks [m]; Robot 2 can reach from -0.2, but
# every box it could pick in the shared area it can also pick later outside of it, without blocking Robot 1
# The fingers close along the belt: a neighbour closer than BLOCK_X in front of or behind a box, in the same lane
# (lateral offset below BLOCK_Y), is hit by them (measured: hit at 0.10 m, free at 0.12 m). Such a box is not picked.
BLOCK_X, BLOCK_Y = 0.11, 0.06
# Shared area: the belt section where the arms can meet, and region B. Each arm plans with the other one
# static, so two arms working there at the same time collide. Only one robot may work in it at a time.
SHARED_X = (-0.3, 0.4)
SHARED_REGION = "region_B"

COLORED = (BoxColor.RED, BoxColor.GREEN, BoxColor.BLUE)
FIXED_ROLES = {1: COLORED, 2: (BoxColor.GRAY,)}                    # Robot 1 sorts, Robot 2 removes gray boxes
SHARED_ROLES = {1: COLORED, 2: (BoxColor.GRAY, BoxColor.GREEN)}    # region B is reachable by both: either sorts green


def uses_shared_area(pick_x: float, region: str) -> bool:
    return region == SHARED_REGION or SHARED_X[0] < pick_x < SHARED_X[1]


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

    Priority in the shared area (`uses_shared_area`): a robot working there keeps it until its task ends
    (`shared_area_busy`); if both free robots want it, the more urgent box wins. The other robot waits: it
    may only take a box outside the shared area. Refused pairs and waiting robots are recorded in `log`.
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
                    and world.time - track.timestamp <= MAX_TRACK_GAP and x_min <= x <= x_max):
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
            if uses_shared_area(position[0], region):
                if shared_area_busy:
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
