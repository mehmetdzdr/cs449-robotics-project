"""Clear-and-sort: a box right in front of or behind a wanted box blocks the fingers, which close along the belt.

Kinematic backend (`plan_clear_and_sort`): a gray box in the way of Robot 1 is taken away first by Robot 2
(grasping it across the belt), then Robot 1 sorts.

Physics backend (`plan_clearing_job`, #101): the blockers to remove come from `planning.clearing.plan_clearing`
(#100), each removal goes to whichever robot can reach that blocker in time, and a robot that cannot take the
blocker to its own region (Robot 1 and a gray box) puts it into the buffer, from where another robot takes it on.
The steps run one after another, never two arms at the cluster at once (see `plan_clear_and_sort` for why).
"""
from dataclasses import dataclass, field

import numpy as np

from conveyor_sort.coordination.allocation import BLOCK_X, BLOCK_Y, LEAD_TIME, MIN_TRACK_AGE, max_track_gap
from conveyor_sort.interfaces import Assignment, BoxColor, Track
from conveyor_sort.planning.access import grasp_access
from conveyor_sort.planning.clearing import plan_clearing
from conveyor_sort.planning.reach import can_pick, is_reachable
from conveyor_sort.sim.regions import BUFFER, BUFFER_ROBOTS, DEFAULT_TARGET, REGION_ROBOTS
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

    Not on the physics backend (always None there: the pair is left alone). The physical grasp keeps Robot 2 at
    the belt 2.2 s longer, and both arms work at the same spot: with CLEAR_DELAY Robot 2 lifted into Robot 1's
    path (7 cm arm penetration), with 2.2 or 3.2 s more Robot 1 could not plan its pick past Robot 2's arm. Clearing
    under physics: `plan_clearing_job`.
    """
    if world.bot is not None:
        return None
    usable = _usable(world, first_seen, blocked)

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


def _usable(world: World, first_seen: dict[int, float], blocked: dict[int, float]):
    """A track that may be planned for now: not blocked, followed long enough, seen recently, moving with the belt."""
    def usable(track: Track) -> bool:
        return (world.time >= blocked.get(track.track_id, 0.0) and world.time - first_seen[track.track_id] >= MIN_TRACK_AGE
                and world.time - track.timestamp <= max_track_gap(world) and track.velocity[0] > 0)
    return usable


# Physics backend: a clearing job's steps run one after another. A physical pick-and-place keeps a robot busy about
# 9 s after the grasp (squeeze, lift, place, release, retreat, home); the next step is decided then and grasps
# LEAD_TIME later. The schedule is checked with this time between two grasps [s].
STEP_TIME = 12.5
NEAR = 0.25            # neighbours considered for clearing: centre within this of the target, along and across [m]
SEARCH_STEP = 2.0      # a clearing that cannot start now is tried with starts this far apart (#102) [s]


@dataclass
class ClearingStep:
    robot_id: int
    track_id: int
    region: str            # where the box goes: its own region, or BUFFER
    across_belt: bool      # grasp axis (see plan_moving_pick)


@dataclass
class ClearingJob:
    target: int                  # track id of the box the clearing is for
    steps: list[ClearingStep]    # the removals in order, then the target (last)


def destination(robot_id: int, color: BoxColor) -> str | None:
    """Where `robot_id` takes a removed box: its own region if the robot reaches it, else the buffer."""
    region = DEFAULT_TARGET[color]
    if robot_id in REGION_ROBOTS[region]:
        return region
    return BUFFER if robot_id in BUFFER_ROBOTS else None


@dataclass
class ClearingDecision:
    job: ClearingJob | None = None                                # to start now
    later: dict[int, float] = field(default_factory=dict)         # target -> earliest start of a feasible clearing [s]
    infeasible: list[int] = field(default_factory=list)           # targets that cannot be cleared in time at all


def _schedule(world: World, target: Track, near: list[Track], plan, roles: dict[int, tuple[BoxColor, ...]],
              buffer_free: int, start: float) -> list[ClearingStep] | None:
    """The steps of a clearing whose first step is decided at `start` (step k grasps at start + LEAD_TIME +
    k * STEP_TIME), or None if the target or a blocker would be out of reach at its time."""
    at = target.predict(start + LEAD_TIME + len(plan.removals) * STEP_TIME)
    sorter = next((r for r in sorted(roles) if target.color in roles[r] and can_pick(world.C, r, at)), None)
    if sorter is None:                                     # checked first: it is the last step, the likeliest to fail
        return None
    steps, in_buffer = [], 0
    for k, removal in enumerate(plan.removals):
        box = near[removal.box]
        at = box.predict(start + LEAD_TIME + k * STEP_TIME)
        options = [(robot_id, destination(robot_id, box.color)) for robot_id in sorted(roles)]
        options = sorted([o for o in options if o[1] is not None], key=lambda o: o[1] == BUFFER)
        for robot_id, region in options:
            if (region != BUFFER or in_buffer < buffer_free) and can_pick(world.C, robot_id, at):
                steps.append(ClearingStep(robot_id, box.track_id, region, removal.axis == "across"))
                in_buffer += region == BUFFER
                break
        else:
            return None                                    # this blocker cannot be removed in time
    return steps + [ClearingStep(sorter, target.track_id, DEFAULT_TARGET[target.color], plan.target_axis == "across")]


def plan_clearing_job(world: World, tracks: list[Track], first_seen: dict[int, float], blocked: dict[int, float],
                      roles: dict[int, tuple[BoxColor, ...]], buffer_free: int) -> ClearingDecision:
    """Clearing for the boxes of a color in `roles` that `allocate` leaves alone because a neighbour blocks their
    grasp along the belt, most urgent (furthest downstream) first.

    `plan_clearing` gives the removals (none if the target can be grasped across the belt). Step k grasps at
    about start + LEAD_TIME + k * STEP_TIME. Each removal goes to a robot of `roles` that can pick the blocker then:
    one that takes it to its own region first, else one that puts it into the buffer (at most `buffer_free` boxes).
    The target goes to a robot whose colors include it and that can pick it at its time. Any robot may remove a
    blocker (Robot 2 with fixed roles takes a colored blocker into the buffer); every box involved must be usable.

    The whole schedule is checked before the first removal (#102). The first target that can be cleared starting
    now gets the `job`. A target that cannot, but can with a later start (every SEARCH_STEP until it would leave the
    belt; e.g. a blocker that only Robot 2 reaches, further downstream), goes into `later` with that start time; one
    that cannot be cleared at any start, or whose neighbours cannot be removed in any order, into `infeasible`.
    """
    usable = _usable(world, first_seen, blocked)
    now, decision = world.time, ClearingDecision()
    candidates = [t for t in tracks if usable(t) and any(t.color in colors for colors in roles.values())]
    for target in sorted(candidates, key=lambda t: -t.position[0]):
        position = target.predict(now)
        near = [t for t in tracks if t.track_id != target.track_id
                and np.all(np.abs(t.predict(now)[:2] - position[:2]) < NEAR)]
        boxes = [(t.color, t.predict(now)) for t in near]
        if not grasp_access(position, boxes).blockers["along"] or not all(usable(t) for t in near):
            continue
        plan = plan_clearing(position, boxes)
        if plan is None:
            decision.infeasible.append(target.track_id)
            continue
        steps = _schedule(world, target, near, plan, roles, buffer_free, now)
        if steps:
            decision.job = ClearingJob(target.track_id, steps)
            return decision
        start = now + SEARCH_STEP
        while target.predict(start + LEAD_TIME)[0] < world.conveyor.geometry.x_end:
            if _schedule(world, target, near, plan, roles, buffer_free, start):
                decision.later[target.track_id] = start
                break
            start += SEARCH_STEP
        else:
            decision.infeasible.append(target.track_id)
    return decision
