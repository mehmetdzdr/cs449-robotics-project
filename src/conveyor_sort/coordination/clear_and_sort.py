"""Clear-and-sort: a gray box right in front of or behind a wanted box blocks Robot 1's fingers, which close
along the belt. Robot 2 takes the gray box away first (grasping it across the belt), then Robot 1 sorts."""
from conveyor_sort.coordination.allocation import BLOCK_X, BLOCK_Y, LEAD_TIME, MAX_TRACK_GAP, MIN_TRACK_AGE
from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.planning.reach import can_pick, is_reachable
from conveyor_sort.sim.regions import DEFAULT_TARGET
from conveyor_sort.sim.world import World

# Belt positions where Robot 2 may take the blocker, tried in this order [m]. They lie in the overlap of the
# workspaces: Robot 2 must reach the gray box and Robot 1, a little later, the wanted box. Further downstream is
# easier for Robot 2 (further from the limit of its reach), further upstream for Robot 1.
CLEAR_XS = (0.1, 0.05, 0.0, -0.05, -0.1)
ROBOT_2_MARGIN = [-0.1, 0.0, 0.15]   # Robot 2 must also reach this far upstream and up: at the limit the lift fails
# Limits found by experiment. Beyond them Robot 2's arm is stretched so low over the belt that it touches the
# wanted box, its lift cannot be planned and it is left holding the gray box in Robot 1's way.
MAX_SPEED = 0.08      # belt speed [m/s]
MAX_LATERAL = 0.01    # offset of the gray box from the belt centre line, away from the robots [m]: near half only
CLEAR_DELAY = 3.0     # Robot 1 grasps this long after Robot 2: the gray box is lifted and on its way to the bin [s]


def blockers(track: Track, tracks: list[Track], time: float) -> list[Track]:
    """Tracks of the boxes that are in the way of a grasp of `track` along the belt at `time`."""
    position = track.predict(time)
    result = []
    for other in tracks:
        offset = other.predict(time) - position
        if other.track_id != track.track_id and abs(offset[0]) < BLOCK_X and abs(offset[1]) < BLOCK_Y:
            result.append(other)
    return result


def plan_clear_and_sort(world: World, tracks: list[Track], first_seen: dict[int, float], blocked: dict[int, float],
                        colors: tuple[BoxColor, ...]) -> tuple[Assignment, Assignment] | None:
    """Assignments (Robot 2: gray blocker to the bin, Robot 1: wanted box to its region) for the first wanted box
    that is blocked by exactly one gray box and can still be cleared in time; None if there is none.

    Robot 2 grasps the gray box at the first of the positions `CLEAR_XS` that works for both robots,
    Robot 1 the wanted box `CLEAR_DELAY` later.
    """
    def usable(track: Track) -> bool:
        return (world.time >= blocked.get(track.track_id, 0.0) and world.time - first_seen[track.track_id] >= MIN_TRACK_AGE
                and world.time - track.timestamp <= MAX_TRACK_GAP and track.velocity[0] > 0)

    for wanted in tracks:
        if wanted.color not in colors or not usable(wanted):
            continue
        in_the_way = blockers(wanted, tracks, world.time)
        if len(in_the_way) != 1 or in_the_way[0].color != BoxColor.GRAY or not usable(in_the_way[0]):
            continue
        gray = in_the_way[0]
        if gray.velocity[0] > MAX_SPEED or gray.position[1] - world.conveyor.geometry.y > MAX_LATERAL:
            continue
        for clear_x in CLEAR_XS:
            clear_time = gray.timestamp + (clear_x - gray.position[0]) / gray.velocity[0]
            pick_time = clear_time + CLEAR_DELAY
            if clear_time < world.time + LEAD_TIME:                # too late: the gray box is already further downstream
                break
            gray_position, wanted_position = gray.predict(clear_time), wanted.predict(pick_time)
            if (can_pick(world.C, 1, wanted_position)
                    and is_reachable(world.C, 2, gray_position) and is_reachable(world.C, 2, gray_position + ROBOT_2_MARGIN)):
                return (Assignment(2, gray.track_id, DEFAULT_TARGET[BoxColor.GRAY], clear_time, gray_position),
                        Assignment(1, wanted.track_id, DEFAULT_TARGET[wanted.color], pick_time, wanted_position))
    return None
