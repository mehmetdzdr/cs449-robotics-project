"""Covered boxes (#177): Robot 2 lifts the cover off a box and puts it aside, then Robot 1 sorts the box.

A two-step job, like a clearing job (#101, `clear_and_sort`): the whole schedule is checked before the first step
(#102), the robots are reserved for it, and one arm is over the belt at a time (`at_belt`). The steps:

1. Robot 2 (`LIFTER`) lifts the cover (`planning.cover_task.CoverTask`). Only Robot 2: the box under a cover is
   colored, only Robot 1 reaches the colored regions, and the cover goes to Robot 2's side of the belt, away from
   Robot 1's part of it. It starts when Robot 2 can pick the knob LEAD_TIME from now and Robot 1 can still pick the
   box HANDOVER_TIME after that (from the grasp of the cover to the grasp of the box: squeeze, lift, the box tracked
   for MIN_TRACK_AGE, LEAD_TIME); a cover that is too far downstream for that is reported (`CoverStats.infeasible`)
   and left alone. While a cover is due (Robot 2 can pick it now or within RESERVE_TIME) no other task starts, so
   that the belt is free for it: a covered box goes first.
2. Robot 1 (`SORTER`) is reserved from the start of step 1. Once the cover is lifted the box is seen (its color is
   known only now); its track (the one where the box is expected, moved on by the belt since the lift) becomes a
   normal `Assignment` for Robot 1, started by the sorting loop like any other, with its retries.

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
RESERVE_TIME = 5.0      # a cover Robot 2 can pick within this is due: no other task starts [s]
CHECK_EVERY = 0.5       # cover decisions (KOMO reachability checks) at most this often [s]
RETRY_DELAY = 0.5       # a cover task that could not be planned is tried again after this [s]
MAX_ATTEMPTS = 3        # cover tasks per cover track
UNCOVERED_MATCH = 0.1   # the uncovered box's track: first seen after the grasp of the cover, the nearest one within
#                         this (xy) of where the box is expected (it has 1.5 cm play in the cover, and it is free to
#                         move on from some time during the lift) [m]
UNCOVERED_TIMEOUT = 8.0   # ... found within this after the lift, or Robot 1 is released [s]


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
        self._due: tuple[str, int | None] = ("none", None)       # latest decision: ("start" | "wait" | "none", track)
        self._next_check = 0.0

    def update(self) -> None:
        """Every step: follow the covers; record any contact of Robot 1 with a cover."""
        world = self.world
        self.tracks = self.tracker.update(self.detect_covers(), world.time)
        for track in self.tracks:
            self.first_seen.setdefault(track.track_id, world.time)
        world.C.computeCollisions()
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

    def decide(self) -> tuple[str, Track | None]:
        """("start", track): Robot 2 can lift this cover now (pick LEAD_TIME from now) and the box can be sorted
        after; ("wait", track): it can within RESERVE_TIME; ("none", None) otherwise. Covers that are too far
        downstream for the whole job are reported in `stats.infeasible` and left alone."""
        world, now = self.world, self.world.time
        due = ("none", None)
        for track in sorted([t for t in self.tracks if self._usable(t)], key=lambda t: -t.position[0]):
            pick_time = now + LEAD_TIME
            if not self.box_reachable(track, pick_time):
                if track.predict(pick_time + HANDOVER_TIME)[0] > 0.0:   # the box would be past Robot 1 (at x = 0):
                    self.blocked[track.track_id] = float("inf")          # only further later
                    self.stats.infeasible += 1
                continue
            if can_pick(world.C, LIFTER, track.predict(pick_time)):
                return "start", track
            if due[0] == "none" and can_pick(world.C, LIFTER, track.predict(pick_time + RESERVE_TIME)):
                due = ("wait", track)
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
        """The running cover task has lifted its cover: the box under it is Robot 1's next task."""
        if self._task is not None and self._task.result.lifted_at is not None and not self._lift_noted:
            self._lift_noted = True
            self.stats.lifted += 1
            result = self._task.result
            self._handover = (result.lifted_from, result.lift_started, result.grasped_at)

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
        if self._task is not None:                     # before the lift: Robot 1 waits for the box
            return [r for r in free if r != SORTER], []
        if now >= self._next_check:
            self._next_check = now + CHECK_EVERY
            kind, track = self.decide()
            self._due = (kind, track.track_id if track else None)
        kind, track_id = self._due
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
            self._due = ("none", None)
            return [r for r in free if r not in (LIFTER, SORTER)], []
        return [], []                                  # a cover is due: nothing else starts meanwhile

    def finished(self, task) -> bool:
        """Called by the sorting loop for every task that ends: True (and recorded) if it was a cover task."""
        from conveyor_sort.planning.cover_task import CoverTask
        if not isinstance(task, CoverTask):
            return False
        result, track_id = task.result, task.assignment.track_id
        self._note_lift()                           # lifted, and ended before the next start()
        self._task = None
        self.results.append(result)
        if result.success:
            self.stats.put_aside += 1
            return True
        self.stats.failures[result.failed_stage] += 1
        if result.lifted_at is None and self.attempts[track_id] < MAX_ATTEMPTS:   # the cover is (still or again)
            self.blocked[track_id] = self.world.time + RETRY_DELAY                # on the belt: tried again
            if result.failed_stage == "pick":       # the arm has not moved: no attempt used
                self.attempts[track_id] -= 1
        return True
