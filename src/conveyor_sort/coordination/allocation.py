"""Task allocation: which free robot picks which tracked box next."""
import numpy as np

from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.planning.reach import is_reachable
from conveyor_sort.sim.regions import DEFAULT_TARGET
from conveyor_sort.sim.world import World

LEAD_TIME = 3.0        # time from the decision to the grasp [s]
MIN_TRACK_AGE = 1.0    # a track must have been followed this long before it is picked (velocity estimate settled) [s]
MAX_TRACK_GAP = 0.2    # ... and must have been seen this recently [s]
REACH_MARGIN = [0.1, 0.0, 0.15]   # safety margin of the pick position: downstream and up (lift height) [m]
# part of the belt (x) where each robot picks. Robot 2 starts downstream of Robot 1's reach (x <= 0.2) with
# clearance: each arm plans with the other one static, picking in the overlap gave arm-arm collisions
PICK_X = {1: (-1.1, 0.2), 2: (0.4, 1.1)}

COLORED = (BoxColor.RED, BoxColor.GREEN, BoxColor.BLUE)
FIXED_ROLES = {1: COLORED, 2: (BoxColor.GRAY,)}                    # Robot 1 sorts, Robot 2 removes gray boxes
SHARED_ROLES = {1: COLORED, 2: (BoxColor.GRAY, BoxColor.GREEN)}    # region B is reachable by both: either sorts green


def allocate(world: World, tracks: list[Track], first_seen: dict[int, float], blocked: dict[int, float],
             roles: dict[int, tuple[BoxColor, ...]], free_robots: list[int],
             busy_regions: set[str] = frozenset()) -> list[Assignment]:
    """At most one Assignment per free robot; no box is given to two robots.

    A robot can take a box if the box has one of the robot's colors (`roles`), has been tracked for a while
    (`first_seen`: track id -> time), is not `blocked` (track id -> not before this time) and is reachable at
    the pick time, `LEAD_TIME` from now. A region holds one box: nothing is assigned to a region in
    `busy_regions` (the other robot is on its way there), and one call never sends two boxes to one region.
    Of all such (robot, box) pairs the most urgent one is served first: the box with the least time left
    before it leaves that robot's part of the belt.
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
    for _, robot_id, track in sorted(pairs, key=lambda pair: pair[0]):
        region = DEFAULT_TARGET[track.color]
        if robot_id in assignments or track.track_id in taken or region in regions:
            continue
        position = track.predict(pick_time)
        # at the edge of the workspace a box can be reached but not lifted: also require a point
        # further downstream and at lift height to be reachable
        if is_reachable(world.C, robot_id, position) and is_reachable(world.C, robot_id, position + REACH_MARGIN):
            assignments[robot_id] = Assignment(robot_id, track.track_id, region, pick_time, position)
            taken.add(track.track_id)
            regions.add(region)
    return list(assignments.values())
