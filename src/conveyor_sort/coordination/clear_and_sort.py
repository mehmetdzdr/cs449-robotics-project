"""Clear-and-sort: a box right in front of or behind a wanted box blocks the fingers, which close along the belt.

`plan_clearing_job` (#101): the blockers to remove come from `planning.clearing.plan_clearing` (#100), each removal
goes to a robot that can reach that blocker in time and owns its color, then the target is sorted. The steps run one
after another, never two arms at the cluster at once. (The kinematic clear-and-sort pair, two arms 4 s apart, was
removed with the kinematic grasp in #113.)
"""
from dataclasses import dataclass, field

import numpy as np

from conveyor_sort.coordination.allocation import (BLOCK_X, BLOCK_Y, LEAD_TIME, MAX_PICK_SPEED, MIN_TRACK_AGE,
                                                   max_track_gap)
from conveyor_sort.interfaces import BoxColor, Track
from conveyor_sort.planning.access import grasp_access
from conveyor_sort.planning.clearing import plan_clearing
from conveyor_sort.planning.reach import can_pick
from conveyor_sort.sim.regions import BUFFER, BUFFER_ROBOTS, DEFAULT_TARGET, REGION_ROBOTS
from conveyor_sort.sim.world import World


def blockers(track: Track, tracks: list[Track], time: float) -> list[Track]:
    """Tracks of the boxes that are in the way of a grasp of `track` along the belt at `time`."""
    position = track.predict(time)
    result = []
    for other in tracks:
        offset = other.predict(time) - position
        if other.track_id != track.track_id and abs(offset[0]) < BLOCK_X and abs(offset[1]) < BLOCK_Y:
            result.append(other)
    return result


def _usable(world: World, first_seen: dict[int, float], blocked: dict[int, float]):
    """A track that may be planned for now: not blocked, followed long enough, seen recently, moving with the belt."""
    def usable(track: Track) -> bool:
        return (world.time >= blocked.get(track.track_id, 0.0) and world.time - first_seen[track.track_id] >= MIN_TRACK_AGE
                and world.time - track.timestamp <= max_track_gap(world) and 0 < track.velocity[0] <= MAX_PICK_SPEED)
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
