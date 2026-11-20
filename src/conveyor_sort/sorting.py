"""Autonomous sorting loop: detect, track, allocate boxes to free robots, pick and place them, repeat."""
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from conveyor_sort.coordination.allocation import (COLORED, FIXED_ROLES, LEAD_TIME, CoordinationLog, allocate,
                                                    uses_shared_area)
from conveyor_sort.coordination.clear_and_sort import plan_clear_and_sort, plan_clearing_job
from conveyor_sort.interfaces import Assignment, BoxColor, Detection, Track
from conveyor_sort.physics.contacts import arm_proximity
from conveyor_sort.planning.pick_place import PickPlaceTask
from conveyor_sort.planning.reach import can_pick
from conveyor_sort.prediction.tracker import Tracker
from conveyor_sort.sim.regions import (BUFFER, DEFAULT_TARGET, REGION_ROBOTS, free_slot, free_slots,
                                       region_contains)
from conveyor_sort.sim.world import World

RETRY_DELAY = 0.5      # a box whose pick could not be planned is tried again after this time [s]
MAX_ATTEMPTS = 3       # a box back on the belt after a failed grasp, slip or drop is tried again, this often in all
RETRY_STAGES = ("grasp", "slip", "lift", "drop")   # failures after which the box may lie on the belt again
FINISH_TIMEOUT = 20.0  # physics backend: at the end of a run, running tasks get this long to finish [s]
TRACK_MATCH = 0.05     # tracking error: a track is compared with the true box nearest to it, if within this [m]
INHERIT_X, INHERIT_Y = 0.15, 0.06   # a new track this close to where a failed box is expected takes over its count [m]


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
    dropped_on: str | None = None   # physics backend: where a slipped / dropped box came to rest


@dataclass
class SortingStats:
    events: list[SortEvent] = field(default_factory=list)
    missed: int = 0            # boxes that left the belt end (any color)
    conflicts: int = 0         # (robot, box) pairs refused at least once because the other robot held the shared area
    waiting_time: float = 0.0  # robot time spent idle only because the shared area was occupied [s]
    clear_and_sort: list[bool] = field(default_factory=list)   # one entry per attempt: wanted box sorted after the gray one was removed
    infeasible: list[tuple[float, BoxColor]] = field(default_factory=list)   # physics: blocked boxes that cannot be
    # cleared in time, reported (time, color) and left alone (#102)
    min_arm_distance: float = float("inf")   # physics backend: closest approach of the two arms [m] (inf: never within 0.2 m)
    arm_contacts: list[tuple[float, str, str]] = field(default_factory=list)   # physics backend: (time, frame, frame)
    tracking_errors: list[float] = field(default_factory=list)   # per step and track: distance to its true box [m]
    arm_waits: int = 0         # physics backend: an arm waited for the other one to finish a motion (#141)
    regions_emptied: int = 0   # physics backend: full regions whose boxes were collected before the next place (#142)

    @property
    def sorted_correctly(self) -> int:
        return sum(e.correct for e in self.events)

    @property
    def failed(self) -> int:
        return sum(not e.success for e in self.events)

    @property
    def failures(self) -> Counter:
        """Failed attempts per stage ("grasp", "slip", "drop", "place", ...)."""
        return Counter(e.failed_stage for e in self.events if not e.success)

    @property
    def drops(self) -> Counter:
        """Where slipped and dropped boxes came to rest ("belt", "table", "box", "region")."""
        return Counter(e.dropped_on for e in self.events if e.dropped_on)


def collect_if_full(world: World, region: str) -> bool:
    """Physics backend: if every slot of `region` is taken, its boxes are collected (taken out, as an operator empties
    a full bin) so that the next box can be placed. Called when a task for the region starts: only one robot works
    in a region at a time, so no arm is above it. Returns whether the region was emptied. Never the buffer (#101):
    its boxes are still to be sorted."""
    conveyor = world.conveyor
    if world.bot is None or region == BUFFER or free_slot(world.C, region, conveyor.colors) is not None:
        return False
    on_belt = set(conveyor.on_belt())
    for box in [b for b in conveyor.colors if b not in on_belt and b not in conveyor.held
                and region_contains(world.C, region, world.C.getFrame(b).getPosition())]:
        conveyor.remove(box)
    return True


def run_robots(world: World, detect: Callable[[], list[Detection]], duration: float,
               roles: dict[int, tuple[BoxColor, ...]] = FIXED_ROLES,
               on_step: Callable[[World], None] | None = None, clear_and_sort: bool = False) -> SortingStats:
    """Every robot in `roles` (robot_id -> colors) sorts boxes of its colors into their default regions for
    `duration` seconds of simulation time. The robots work at the same time; `allocate` decides who takes
    which box (with `SHARED_ROLES` green boxes go to whichever robot is free). A box with a neighbour right in
    front of or behind it is left alone, except with `clear_and_sort`. Kinematic backend: if the neighbour is gray,
    Robot 2 removes it first and Robot 1 then sorts the box (needs both robots). Physics backend (#101): a clearing
    job (`plan_clearing_job`, planned while every robot is free) removes the blockers one by one, each by whichever
    robot can reach it, into its own region or the buffer, then the target is sorted; a robot with nothing else to
    do takes a box from the buffer to its region. `SortingStats.clear_and_sort` has one entry per job; a blocked box that
    cannot be cleared before it leaves reach is reported in `SortingStats.infeasible` instead of tried (#102).

    `detect()` returns the detections of the current camera frame (ground truth or a vision detector).

    Recovery: a box whose attempt failed after the arm moved (`RETRY_STAGES`) is tried again RETRY_DELAY later
    if it is back on the belt and still reachable, up to MAX_ATTEMPTS attempts. `SortingStats.failures` counts
    the failures per stage, `SortingStats.drops` where slipped or dropped boxes came to rest. Physics backend:
    tasks still running at the end are finished (at most FINISH_TIMEOUT more), so no arm keeps a box.
    """
    tracker = Tracker()
    stats = SortingStats()
    first_seen: dict[int, float] = {}
    blocked: dict[int, float] = {}         # track id -> not before this time; forever once the arm has moved for it
    tracks: list[Track] = []
    last_placed: dict[str, str] = {}       # region -> box lying in it
    tasks: dict[int, tuple[PickPlaceTask, Track]] = {}
    attempts: Counter = Counter()          # track id -> pick-and-place tasks started for it
    failed_boxes: list = []                # (pick position, pick time, attempts) of boxes that may be back on the belt

    def inherit(track: Track) -> None:
        """A new track where a failed box is expected (its pick position moved on by the belt since then, give or
        take the time it was held) takes over that box's attempt count, so a box is not tried endlessly."""
        speed = world.conveyor.config.speed
        for i, (position, pick_time, count) in enumerate(failed_boxes):
            expected_x = position[0] + speed * (world.time - pick_time)
            if abs(track.position[0] - expected_x) < INHERIT_X and abs(track.position[1] - position[1]) < INHERIT_Y:
                attempts[track.track_id] = max(attempts[track.track_id], count)
                if count >= MAX_ATTEMPTS:
                    blocked[track.track_id] = float("inf")
                del failed_boxes[i]
                return
    scheduled: dict[int, tuple[float, Assignment]] = {}   # clear-and-sort: robot -> (time to start moving, assignment)
    pair: dict = {}                        # the running clear-and-sort: track ids "gray" and "wanted", "gray_ok"
    clearing: dict = {}                    # physics: the running clearing job: "job", "next" (index of the next step)
    buffer: list[str] = []                 # physics: boxes in the buffer, in the order they were put there
    not_before = {"clearing": 0.0, "buffer": 0.0}   # physics: next time a clearing job / a buffer pick is tried
    log = CoordinationLog()
    end_time = world.time + duration
    missed_before = world.conveyor.missed

    def job_tracks() -> set[int]:
        return {step.track_id for step in clearing["job"].steps} if clearing else set()

    def finish(robot_id: int) -> None:
        task, track = tasks.pop(robot_id)
        result, region = task.result, task.assignment.target_region
        stats.arm_waits += task.waits
        if task.box is not None and result.failed_stage in ("pick", "full"):   # buffer pick, the arm has not moved:
            buffer.append(task.box)                                             # the box is still there, later
            not_before["buffer"] = world.time + RETRY_DELAY
            return
        if (task.box is not None and not result.success and attempts[track.track_id] < MAX_ATTEMPTS
                and region_contains(world.C, BUFFER, world.C.getFrame(task.box).getPosition())):
            buffer.append(task.box)            # fell back into the buffer: tried again
            not_before["buffer"] = world.time + RETRY_DELAY
        in_job = track.track_id in job_tracks() or track.track_id in pair.values() or task.box is not None
        if result.failed_stage == "pick" and not in_job:   # no plan from here (e.g. box just
            blocked[track.track_id] = world.time + RETRY_DELAY   # entering the workspace): no time was used, try again later
            attempts[track.track_id] -= 1
            return
        if result.failed_stage in RETRY_STAGES and not in_job:
            if attempts[track.track_id] < MAX_ATTEMPTS:
                blocked[track.track_id] = world.time + RETRY_DELAY   # picked again if it is (still or again) on the belt
            # while it was held the box was not seen and its track may have been dropped: a new track where the box
            # fell back carries on the count (see `inherit`)
            failed_boxes.append((task.assignment.pick_position, task.assignment.pick_time, attempts[track.track_id]))
        correct = (result.success and DEFAULT_TARGET[world.conveyor.colors[result.box]] == region
                   and region_contains(world.C, region, world.C.getFrame(result.box).getPosition()))
        if result.success:
            if world.bot is None:              # physics: regions hold several boxes (slots, #93)
                last_placed[region] = result.box
        stats.events.append(SortEvent(world.time, track.color, region, result.success, correct,
                                      result.failed_stage, result.planning_time, robot_id, result.dropped_on))
        if track.track_id == pair.get("gray"):
            pair["gray_ok"] = result.success
            if not result.success and scheduled.pop(1, None):    # blocker still there: Robot 1 does not start
                stats.clear_and_sort.append(False)
                pair.clear()
        elif track.track_id == pair.get("wanted"):
            stats.clear_and_sort.append(correct and pair["gray_ok"])
            pair.clear()
        elif track.track_id in job_tracks():
            if not result.success or track.track_id == clearing["job"].target:   # a failed removal ends the job:
                stats.clear_and_sort.append(correct)                            # the target is left alone
                clearing.clear()
            elif region == BUFFER:
                buffer.append(result.box)

    def start(assignment: Assignment, across_belt: bool = False) -> None:
        track = by_id[assignment.track_id]
        if assignment.target_region in last_placed:   # regions are emptied (box collected) before the next box arrives
            world.conveyor.remove(last_placed.pop(assignment.target_region))
        stats.regions_emptied += collect_if_full(world, assignment.target_region)   # physics: only when full (#142)
        tasks[assignment.robot_id] = PickPlaceTask(world, assignment, track, across_belt), track
        attempts[assignment.track_id] += 1

    def shared_area_busy() -> bool:
        return any(uses_shared_area(task.assignment.pick_position[0], task.assignment.target_region)
                   for task, _ in tasks.values())

    def run_clearing(free: list[int]) -> list[int]:
        """Physics backend: plan a clearing job (only while every robot is free) or start its next step once the
        previous one has ended. Returns the free robots that are not reserved for the job."""
        if not clearing and set(free) == set(roles) and world.time >= not_before["clearing"]:
            not_before["clearing"] = world.time + RETRY_DELAY      # KOMO reachability checks: not every step
            decision = plan_clearing_job(world, tracks, first_seen, blocked, roles,
                                         len(free_slots(world.C, BUFFER, world.conveyor.colors)))
            colors = {track.track_id: track.color for track in tracks}
            for track_id in decision.infeasible:          # reported once, then left alone (#102)
                blocked[track_id] = float("inf")
                stats.infeasible.append((world.time, colors[track_id]))
            for track_id, earliest in decision.later.items():   # planned again then (not `start`: that is the
                blocked[track_id] = earliest                    # function that starts a task, below)
            if decision.job:
                clearing.update(job=decision.job, next=0)
                for step in decision.job.steps:
                    blocked[step.track_id] = float("inf")
        if not clearing:
            return free
        job, k = clearing["job"], clearing["next"]
        running = any(task.assignment.track_id in job_tracks() for task, _ in tasks.values())
        if not running and k < len(job.steps) and job.steps[k].robot_id in free:
            step = job.steps[k]
            track, pick_time = by_id.get(step.track_id), world.time + LEAD_TIME
            if track is None or not can_pick(world.C, step.robot_id, track.predict(pick_time)):
                stats.clear_and_sort.append(False)      # track lost, or out of reach by now (time limits: #102)
                clearing.clear()
                return free
            if not (uses_shared_area(track.predict(pick_time)[0], step.region) and shared_area_busy()):
                start(Assignment(step.robot_id, step.track_id, step.region, pick_time, track.predict(pick_time)),
                      across_belt=step.across_belt)
                clearing["next"] = k + 1
        reserved = {step.robot_id for step in job.steps}
        return [robot_id for robot_id in free if robot_id not in reserved and robot_id not in tasks]

    def empty_buffer(free: list[int]) -> None:
        """Physics backend: a free robot takes the oldest buffer box it can take to that box's own region, only
        while no other task runs (and nothing else starts meanwhile, see the loop): with the other arm working at
        the same time the arms came within 5.5 cm and the place into the bin could not be planned (CI, #139)."""
        if not buffer or world.time < not_before["buffer"] or tasks:
            return
        for robot_id in free:
            box = next((b for b in buffer if robot_id in REGION_ROBOTS[DEFAULT_TARGET[world.conveyor.colors[b]]]), None)
            if box is None:
                continue
            buffer.remove(box)
            color, position = world.conveyor.colors[box], world.C.getFrame(box).getPosition().copy()
            track = Track(-1 - int(box.split("_")[1]), color, position, np.zeros(3), world.time)   # standing still
            assignment = Assignment(robot_id, track.track_id, DEFAULT_TARGET[color], world.time + LEAD_TIME, position)
            stats.regions_emptied += collect_if_full(world, assignment.target_region)
            tasks[robot_id] = PickPlaceTask(world, assignment, track, box=box), track
            attempts[track.track_id] += 1
            return

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
            if any(task.box is not None for task, _ in tasks.values()):   # a box is taken from the buffer: the
                free = []                                                  # other arm waits (see empty_buffer)
            if clear_and_sort and world.bot is not None:
                free = run_clearing(free)
            if clear_and_sort and world.bot is None and not pair and set(free) >= {1, 2}:
                plan = plan_clear_and_sort(world, tracks, first_seen, blocked, roles[1])
                if plan:
                    pair.update(gray=plan[0].track_id, wanted=plan[1].track_id, gray_ok=False)
                    for assignment in plan:
                        blocked[assignment.track_id] = float("inf")
                        scheduled[assignment.robot_id] = assignment.pick_time - LEAD_TIME, assignment
                    free = []
            busy_regions = {task.assignment.target_region for task, _ in tasks.values()}
            for assignment in (allocate(world, tracks, first_seen, blocked, roles, free, busy_regions,
                                        shared_area_busy(), log) if free else []):
                blocked[assignment.track_id] = float("inf")
                start(assignment)
            if world.bot is not None:
                empty_buffer([robot_id for robot_id in free if robot_id not in tasks])
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
        if world.bot is not None:         # arm-arm contacts at the simulated poses (see physics/contacts.py)
            distance, contacts = arm_proximity(world.C)
            stats.min_arm_distance = min(stats.min_arm_distance, distance)
            stats.arm_contacts += [(world.time, a, b) for a, b, _ in contacts]
        tracks[:] = tracker.update(detect(), world.time)
        for task, _ in tasks.values():     # physics: a box displaced during the approach is aimed at again (#104)
            task.observe(tracks)
        truth = [world.C.getFrame(b).getPosition() for b in world.conveyor.on_belt()]   # evaluation only
        for track in tracks:
            error = min((np.linalg.norm(track.predict(world.time) - p) for p in truth), default=np.inf)
            if error < TRACK_MATCH:
                stats.tracking_errors.append(float(error))
        for track in tracks:
            if track.track_id not in first_seen and failed_boxes:
                inherit(track)
            first_seen.setdefault(track.track_id, world.time)
        if on_step:
            on_step(world)
    finish_end = world.time + FINISH_TIMEOUT
    while world.bot is not None and tasks and world.time < finish_end:   # no arm keeps a box after the run
        for robot_id in list(tasks):
            if not tasks[robot_id][0].advance():
                finish(robot_id)
        world.step()
    stats.missed = world.conveyor.missed - missed_before
    stats.conflicts = len(log.conflicts)
    return stats


def run_single_robot(world: World, detect: Callable[[], list[Detection]], duration: float, robot_id: int = 1,
                     colors: tuple[BoxColor, ...] = COLORED,
                     on_step: Callable[[World], None] | None = None) -> SortingStats:
    """Single-robot baseline: only `robot_id` works."""
    return run_robots(world, detect, duration, {robot_id: colors}, on_step)
