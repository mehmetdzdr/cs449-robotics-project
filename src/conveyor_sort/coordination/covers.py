"""Covered boxes (#177): Robot 2 lifts the cover off a box and puts it aside, then Robot 1 sorts the box.

A two-step job, like a clearing job (#101, `clear_and_sort`): the whole schedule is checked before the first step
(#102), and one arm is over the belt at a time (`at_belt`). The steps:

1. Robot 2 (`LIFTER`) lifts the cover (`planning.cover_task.CoverTask`). Only Robot 2: the box under a cover is
   colored, only Robot 1 reaches the colored regions, and the cover goes to Robot 2's side of the belt, away from
   Robot 1's part of it. Each cover has a window (`window`, belt arithmetic): from when its knob is in Robot 2's lift
   range (LIFT_X) at pick time until the box under it would no longer reach Robot 1's part of the belt in time (picked
   HANDOVER_TIME after the cover: squeeze, lift, the box tracked for MIN_TRACK_AGE, LEAD_TIME). Inside the window the
   lift starts as soon as Robot 2 is free, the belt is free and the KOMO checks pass (Robot 2 can pick the knob, Robot
   1 the box). A cover whose window has closed is reported (`CoverStats.infeasible`) and left alone.
   The other tasks go on meanwhile; only when the window would close otherwise is a robot held back: Robot 2 from
   other tasks within LIFTER_TASK_TIME of the window's end, Robot 1 from going to the belt within BELT_TIME of it (a
   pick keeps the belt that long, and Robot 2 could not start).
2. Robot 1 (`SORTER`) is reserved from the end of the lift motion (before, the lifting arm keeps it off the belt
   anyway) until it has the box. The box is seen once the cover is lifted (its color is known only now); its track
   (the one where the box is expected, moved on by the belt since the lift) becomes a normal `Assignment` for Robot
   1, started by the sorting loop like any other, with its retries.

`Uncovering` is switched on by `run_robots(..., covers=Uncovering(world, detect_covers))`; without it the loop is
unchanged. It follows the covers with its own `Tracker` (`perception.covers.get_cover_detector`).
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from conveyor_sort.coordination.allocation import (BLOCK_X, BLOCK_Y, LEAD_TIME, MAX_PICK_SPEED, MIN_TRACK_AGE, PICK_X,
                                                   max_track_gap)
from conveyor_sort.interfaces import Assignment, Detection, Track
from conveyor_sort.physics.grasping import SQUEEZE_SETTLE, SQUEEZE_TIME
from conveyor_sort.planning.pick_place import LIFT_TIME
from conveyor_sort.planning.reach import can_pick
from conveyor_sort.prediction.tracker import Tracker
from conveyor_sort.sim.conveyor import BOX_SIZE
from conveyor_sort.sim.covers import COVER_REST, COVER_SPOT
from conveyor_sort.sim.regions import DEFAULT_TARGET, REGION_ROBOTS
from conveyor_sort.sim.world import World

LIFTER, SORTER = 2, 1
# grasp of the cover -> grasp of the box: squeeze, lift, settle; the box tracked for MIN_TRACK_AGE, then LEAD_TIME;
# plus 0.5 s for the step in which the track becomes usable [s]
HANDOVER_TIME = SQUEEZE_TIME + SQUEEZE_SETTLE + LIFT_TIME + SQUEEZE_SETTLE + MIN_TRACK_AGE + LEAD_TIME + 0.5
LIFT_X = (-0.5, 0.4)    # Robot 2 lifts a knob at x in this range (at pick time): `can_pick` failed at -0.55, lifts in
#                         CI at -0.52 to -0.19; a prefilter, the KOMO check decides [m]
BELT_TIME = LEAD_TIME + SQUEEZE_TIME + SQUEEZE_SETTLE + LIFT_TIME + 0.7   # a pick keeps the belt this long (+ planning) [s]
LIFTER_TASK_TIME = 12.0   # a pick and place of Robot 2 keeps it busy about this long [s]
CHECK_EVERY = 0.5       # cover decisions (KOMO reachability checks) at most this often [s]
RETRY_DELAY = 0.5       # a cover task that could not be planned is tried again after this [s]
MAX_ATTEMPTS = 3        # cover tasks per cover track
UNCOVERED_MATCH = 0.1   # the uncovered box's track: first seen after the grasp of the cover, the nearest one within
#                         this (xy) of where the box is expected (it has 1.5 cm play in the cover, and it is free to
#                         move on from some time during the lift) [m]
UNCOVERED_TIMEOUT = 5.0   # ... found within this after the lift, or Robot 1 is released (usable after about 1.5 s) [s]


@dataclass
class CoverStats:
    lifted: int = 0          # covers lifted off their box
    put_aside: int = 0       # covers standing on the spot afterwards (CoverTask success)
    handed_over: int = 0     # uncovered boxes given to Robot 1
    infeasible: int = 0      # covered boxes that could not be uncovered and sorted in time (left alone)
    lost: int = 0            # uncovered boxes not found or not pickable after the lift (Robot 1 released)
    failures: Counter = field(default_factory=Counter)   # failed cover tasks per stage
    cover_contacts: list[tuple[float, str, str]] = field(default_factory=list)   # (time, Robot 1 frame, cover frame)


class Uncovering:
    def __init__(self, world: World, detect_covers: Callable[[], list[Detection]]):
        self.world = world
        self.detect_covers = detect_covers
        self.tracker = Tracker()
        self.tracks: list[Track] = []               # cover tracks (knob positions)
        self.first_seen: dict[int, float] = {}
        self.blocked: dict[int, float] = {}         # cover track id -> not before this time
        self.attempts: Counter = Counter()
        self.stats = CoverStats()
        self._task = None                           # the running CoverTask
        self._lift_noted = False                    # its lift has started the handover
        self._handover: tuple[np.ndarray, float, float] | None = None   # after a lift: (box position, moving from
        #                                                                   this time, its track seen after this)
        self.results = []                           # CoverResult of every cover task, in order
        self._due: tuple[str, int | None, float] = ("none", None, np.inf)   # latest decision: (kind, track, latest)
        self._next_check = 0.0

    def update(self) -> None:
        """Every step, after the sorting loop's `arm_proximity` (which computes the collisions of this state): follow
        the covers; record any contact of Robot 1 with a cover."""
        world = self.world
        self.tracks = self.tracker.update(self.detect_covers(), world.time)
        for track in self.tracks:
            self.first_seen.setdefault(track.track_id, world.time)
        for a, b, d in world.C.getCollisions(0.01):
            arm, other = (a, b) if a.startswith("l_") else (b, a)
            if d <= 0 and arm.startswith("l_") and other.startswith("cover_"):
                self.stats.cover_contacts.append((world.time, arm, other))

    def _usable(self, track: Track) -> bool:
        world = self.world
        return (world.time >= self.blocked.get(track.track_id, 0.0)
                and world.time - self.first_seen[track.track_id] >= MIN_TRACK_AGE
                and world.time - track.timestamp <= max_track_gap(world) and 0 < track.velocity[0] <= MAX_PICK_SPEED)

    def box_reachable(self, track: Track, pick_time: float) -> bool:
        """Robot 1 can pick the box under this cover HANDOVER_TIME after the cover is grasped at `pick_time`."""
        at = track.predict(pick_time + HANDOVER_TIME) - [0, 0, COVER_REST - BOX_SIZE / 2]
        return PICK_X[SORTER][0] <= at[0] <= PICK_X[SORTER][1] and can_pick(self.world.C, SORTER, at)

    def window(self, track: Track) -> tuple[float, float]:
        """(earliest, latest) time to start lifting this cover (its knob picked LEAD_TIME later): the knob in LIFT_X,
        and the box under it still in Robot 1's part of the belt HANDOVER_TIME after the grasp. Belt arithmetic, no
        KOMO."""
        now = self.world.time
        x, v = track.predict(now)[0], track.velocity[0]
        last_x = min(LIFT_X[1], PICK_X[SORTER][1] - v * HANDOVER_TIME)
        return now + (LIFT_X[0] - x) / v - LEAD_TIME, now + (last_x - x) / v - LEAD_TIME

    def decide(self) -> tuple[str, Track | None, float]:
        """(kind, track, latest start) for the most urgent cover: "start" if it can be lifted now (inside its window,
        Robot 2 can pick the knob LEAD_TIME from now, Robot 1 the box after), "wait" if its window is ahead or the KOMO
        checks failed; ("none", None, inf) without covers. Covers whose window has closed (or never opens: too fast a
        belt) are reported in `stats.infeasible` and left alone."""
        world, now = self.world, self.world.time
        due = ("none", None, np.inf)
        for track in sorted([t for t in self.tracks if self._usable(t)], key=lambda t: -t.position[0]):
            earliest, latest = self.window(track)
            if latest < now or earliest > latest:
                self.blocked[track.track_id] = float("inf")
                self.stats.infeasible += 1
                continue
            if earliest <= now:
                pick_time = now + LEAD_TIME
                if can_pick(world.C, LIFTER, track.predict(pick_time)) and self.box_reachable(track, pick_time):
                    return "start", track, latest
            if due[0] == "none":
                due = ("wait", track, latest)
        return due

    def handover(self, tracks: list[Track], first_seen: dict[int, float]) -> Assignment | None:
        """Robot 1's assignment for the uncovered box once its track is usable, else None. Ends the handover (and
        Robot 1's reservation) when it is given out, when the box cannot be picked, or after UNCOVERED_TIMEOUT."""
        world, now = self.world, self.world.time
        position, since, seen_after = self._handover
        expected = position + self._belt_velocity() * (now - since)
        distance = {t.track_id: np.linalg.norm(t.predict(now)[:2] - expected[:2]) for t in tracks
                    if first_seen.get(t.track_id, -np.inf) >= seen_after}
        near = sorted([t for t in tracks if distance.get(t.track_id, np.inf) < UNCOVERED_MATCH],
                      key=lambda t: distance[t.track_id])
        track = near[0] if near else None
        if track is None or now - first_seen[track.track_id] < MIN_TRACK_AGE or now - track.timestamp > max_track_gap(world):
            if now - since > UNCOVERED_TIMEOUT:
                self.stats.lost += 1
                self._handover = None
            return None
        self._handover = None
        pick_time = now + LEAD_TIME
        at, region = track.predict(pick_time), DEFAULT_TARGET[track.color]
        neighbour = any(o.track_id != track.track_id and abs(o.predict(pick_time)[0] - at[0]) < BLOCK_X
                        and abs(o.predict(pick_time)[1] - at[1]) < BLOCK_Y for o in tracks)
        if SORTER not in REGION_ROBOTS[region] or neighbour or not can_pick(world.C, SORTER, at):
            self.stats.lost += 1                   # left to the normal allocation (e.g. a gray box: Robot 2)
            return None
        self.stats.handed_over += 1
        return Assignment(SORTER, track.track_id, region, pick_time, at)

    def _belt_velocity(self) -> np.ndarray:
        return np.array([self.world.conveyor.config.speed, 0.0, 0.0])

    def _note_lift(self) -> None:
        """The running cover task has ended its lift motion: the box under it is Robot 1's next task (from now on,
        before the loop's allocation can give Robot 1 another box: the belt is free again)."""
        task = self._task
        if task is not None and task.result.lift_started is not None and not task.at_belt and not self._lift_noted:
            self._lift_noted = True
            self._handover = (task.result.lifted_from, task.result.lift_started, task.result.grasped_at)

    def start(self, free: list[int], tasks: dict, shared_area_busy: bool, tracks: list[Track],
              first_seen: dict[int, float]) -> tuple[list[int], list[Assignment]]:
        """Called by the sorting loop before its own allocation: starts a cover task (into `tasks`, like the loop's
        own), returns the robots that stay free for the loop and the assignments to start (the uncovered box)."""
        from conveyor_sort.planning.cover_task import CoverTask
        world, now = self.world, self.world.time
        self._note_lift()
        if self._handover is not None:
            if SORTER not in free or shared_area_busy:
                return [r for r in free if r != SORTER], []
            assignment = self.handover(tracks, first_seen)
            if assignment is not None:
                return [r for r in free if r != SORTER], [assignment]
            return [r for r in free if r != SORTER or self._handover is None], []
        if self._task is not None:                     # before the end of the lift its arm keeps the belt busy
            return free, []
        if now >= self._next_check:
            self._next_check = now + CHECK_EVERY
            kind, track, latest = self.decide()
            self._due = (kind, track.track_id if track else None, latest)
        kind, track_id, latest = self._due
        track = next((t for t in self.tracks if t.track_id == track_id), None)
        if kind == "none" or track is None or not self._usable(track):
            return free, []
        if kind == "start" and LIFTER in free and not shared_area_busy:
            pick_time = now + LEAD_TIME
            self._task = CoverTask(world, Assignment(LIFTER, track.track_id, COVER_SPOT, pick_time,
                                                     track.predict(pick_time)), track)
            self._lift_noted = False
            tasks[LIFTER] = self._task, track
            self.attempts[track.track_id] += 1
            self.blocked[track.track_id] = float("inf")
            self._due = ("none", None, np.inf)
            return [r for r in free if r != LIFTER], []
        held = {LIFTER} if latest - now < LIFTER_TASK_TIME else set()   # only what would close the window
        if latest - now < BELT_TIME:
            held.add(SORTER)
        return [r for r in free if r not in held], []

    def finished(self, task) -> bool:
        """Called by the sorting loop for every task that ends: True (and recorded) if it was a cover task."""
        from conveyor_sort.planning.cover_task import CoverTask
        if not isinstance(task, CoverTask):
            return False
        result, track_id = task.result, task.assignment.track_id
        self._note_lift()                           # lifted, and ended before the next start()
        self._task = None
        self.results.append(result)
        self.stats.lifted += result.lifted_at is not None
        if result.success:
            self.stats.put_aside += 1
            return True
        self.stats.failures[result.failed_stage] += 1
        if result.lifted_at is None and self.attempts[track_id] < MAX_ATTEMPTS:   # the cover is (still or again)
            self.blocked[track_id] = self.world.time + RETRY_DELAY                # on the belt: tried again
            if result.failed_stage == "pick":       # the arm has not moved: no attempt used
                self.attempts[track_id] -= 1
        return True
