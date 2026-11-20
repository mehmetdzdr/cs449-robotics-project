"""Planning interface for the rest of the system: carry out one Assignment (pick a moving box, place it, go home)."""
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from conveyor_sort.interfaces import Assignment, Track
from conveyor_sort.physics.friction_grasp import FINGER_OPEN, FINGER_SQUEEZE
from conveyor_sort.physics.grasping import (SLIP_TOLERANCE, SQUEEZE_SETTLE, SQUEEZE_TIME, box_at_gripper, holding,
                                            ignoring, planning_view)
from conveyor_sort.planning.execute import follow_path, grasp, release
from conveyor_sort.planning.pick import Plan, plan_home, plan_lift, plan_moving_pick, plan_place
from conveyor_sort.sim.conveyor import BOX_SIZE
from conveyor_sort.sim.regions import REGIONS, SLOTS, free_slot, free_slots, region_contains, slot_position
from conveyor_sort.sim.scene import ROBOT_PREFIX
from conveyor_sort.sim.world import World

# durations of the stages after the grasp [s]
LIFT_TIME, PLACE_TIME, RETREAT_TIME, HOME_TIME = 1.0, 3.0, 1.0, 2.0
MAX_TILT = 10.0   # physics backend: a placed box must be upright within this [deg]
DROP_DISTANCE = 0.04   # physics backend: a carried box this far from the gripper centre has been dropped [m]
REPLAN_DISTANCE = 0.015   # physics backend: the box predicted this far from where the pick aims: re-plan [m]
MIN_REPLAN_TIME = 0.5     # ... if at least this much time is left before the grasp [s]
CLEARANCE = 0.02   # physics backend: the arms must stay this far apart while both move [m] (#141)
MAX_WAIT = 15.0    # ... an arm waits at most this long for the other one to finish its motion [s]
LOST_TIME = 1.2           # ... a box not seen for this long has left the belt (BotOp: an image every 0.5 s) [s]


@dataclass
class PickPlaceResult:
    success: bool
    planning_time: float            # summed KOMO solve time of all stages [s]
    failed_stage: str | None        # "pick", "lost", "grasp", "slip", "lift", "place", "drop", "release", "retreat", "home",
                                    # "full", or None on success
    box: str | None                 # frame name of the grasped box
    place_error: float | None = None   # physics backend: placed box centre to its slot centre in xy [m]
    tilt: float | None = None          # physics backend: placed box's tilt from upright [deg]
    dropped_on: str | None = None      # physics backend, after "slip" or "drop": "belt", "table", "box", "region", "floor"


class PickPlaceTask:
    """Carries out one assignment step by step, so that the caller can run both robots at the same time.

    Call `advance()` once before every `world.step()`: it moves this robot's arm (only its own joints) by one
    simulation step and returns False once the task has ended; `result` is final then. The box is grasped at
    `assignment.pick_time` where `track` predicts it. Each stage is planned from the state the previous one ended
    in, with the other arm treated as static at its current pose. A pick is only started if the lift from its
    grasp pose can be planned too (else: failed stage "pick", the arm has not moved). On a later failure the arm
    stops where it is (after a failed grasp it goes home).

    Physics backend (`World(physics=True)`): the grasp is physical. At pick time the belt stops driving the box
    under the gripper, the fingers squeeze it and the box is held by friction only. Failed stage "grasp": the
    fingers closed on nothing (then they open and the arm goes home); "slip": the box did not rise with the
    gripper during the lift. The place is physical too: the box is lowered to its resting height in the first
    free slot of the region (`regions.SLOTS`; `drop` higher, if given), the fingers open and the box is on its own
    while the arm retreats and goes home. `result.place_error` (box centre to slot centre in xy) and
    `result.tilt` are measured at the end; failed stage "release": the box is not upright (tilt >= MAX_TILT) or
    not inside the region; "full": every slot is taken (checked before the arm moves). `pick_only=True` ends
    after the lift (success = the box is held).
    """

    def __init__(self, world: World, assignment: Assignment, track: Track, across_belt: bool = False,
                 pick_only: bool = False, drop: float = 0.0):
        self.assignment = assignment
        self.across_belt = across_belt     # grasp on the box faces at -y / +y (see plan_moving_pick)
        self.pick_only = pick_only         # physics backend: end after the lift (success = the box is held)
        self.drop = drop                   # physics backend: release the box this much higher than resting [m]
        self.result = PickPlaceResult(False, 0.0, None, None)
        self._world, self._track = world, track
        self._picking, self._update, self._replans = False, None, 0   # physics backend: see observe()
        self.waits = 0     # physics backend: stages that waited for the other arm, or picks given up for it (#141)
        self._steps = self._run_physical(world, assignment, track) if world.bot is not None else \
            self._run(world, assignment, track)

    def advance(self) -> bool:
        return next(self._steps, False)

    @staticmethod
    def _box_near(world: World, track: Track) -> str | None:
        """The box on the belt at the track's position (within 3 cm), as the planner sees it in C."""
        boxes = {b: np.linalg.norm(world.C.getFrame(b).getPosition() - track.position) for b in world.conveyor.on_belt()}
        box = min(boxes, key=boxes.get, default=None)
        return box if box is not None and boxes[box] < 0.03 else None

    @property
    def replans(self) -> int:
        """How often the approach was re-planned because the box was displaced (#104)."""
        return self._replans

    def observe(self, tracks: list[Track]) -> None:
        """Physics backend, while the arm approaches the box: compare its current track with the one the pick was
        planned for. Predicted more than REPLAN_DISTANCE away at pick time (pushed by a finger or a neighbour), the
        pick is re-planned from where the arm is, if at least MIN_REPLAN_TIME is left; not seen for LOST_TIME (pushed
        off the belt), the attempt ends with "lost" and the arm goes home."""
        if not self._picking or self._update is not None:
            return
        world, pick_time = self._world, self.assignment.pick_time
        track = next((t for t in tracks if t.track_id == self._track.track_id), None)
        if track is None or world.time - track.timestamp > LOST_TIME:
            self._update = "lost"
        elif (np.linalg.norm(track.predict(pick_time) - self._track.predict(pick_time)) > REPLAN_DISTANCE
              and pick_time - world.time > MIN_REPLAN_TIME):
            self._update, self._track = track, track

    def _run(self, world: World, assignment: Assignment, track: Track) -> Iterator[bool]:
        C, robot, result = world.C, assignment.robot_id, self.result

        def run(stage: str, plan: Plan, duration: float) -> Iterator[bool]:
            result.planning_time += plan.planning_time
            if not plan.feasible:
                result.failed_stage = result.failed_stage or stage
                return False
            for _ in follow_path(world, plan.path, duration, robot):
                yield True
            return True

        if not (yield from run("pick", self._plan_pick(world, assignment, track), assignment.pick_time - world.time)):
            return
        result.box = grasp(world, robot)
        if result.box is None:
            result.failed_stage = "grasp"
            yield from run("home", plan_home(C, robot), HOME_TIME)
            return
        result.success = (
            (yield from run("lift", plan_lift(C, robot), LIFT_TIME))
            and (yield from run("place", plan_place(C, robot, result.box, assignment.target_region), PLACE_TIME))
            and (release(world, result.box, assignment.target_region) or True)
            and (yield from run("retreat", plan_lift(C, robot, height=0.1), RETREAT_TIME))
            and (yield from run("home", plan_home(C, robot), HOME_TIME))
        )


    def _plan_pick(self, world: World, assignment: Assignment, track: Track) -> Plan:
        """Moving pick, feasible only if the lift from its grasp pose can be planned too."""
        C, robot = world.C, assignment.robot_id
        pick = plan_moving_pick(C, robot, track, world.time, assignment.pick_time, across_belt=self.across_belt)
        if pick.feasible:      # from some grasp poses the arm cannot lift and would be left holding the box:
            q = C.getJointState()                    # check the lift from the pose the pick ends in, before moving
            C.setJointState(pick.path[-1])
            lift = plan_lift(C, robot)
            C.setJointState(q)
            pick = Plan(pick.path, lift.feasible, pick.planning_time + lift.planning_time)
        return pick

    def _run_physical(self, world: World, assignment: Assignment, track: Track) -> Iterator[bool]:
        C, robot, result = world.C, assignment.robot_id, self.result

        others = [r for r in world.fingers if r != robot]

        def too_close(path, duration: float) -> bool:
            """The other arm moves while this one follows `path` and they come closer than CLEARANCE (#141)."""
            now = world.time + world._t0
            return (any(world.arms.moving_until(r) > now for r in others)
                    and world.arms.clearance(C, robot, path, duration) < CLEARANCE)

        def run(stage: str, plan_fn, duration: float, carrying: str | None = None) -> Iterator[bool]:
            """Plan the stage (`plan_fn()`, from the current state) and follow the plan; with `carrying`, stop as soon
            as that box is no longer at the gripper ("drop"). Each arm is planned with the other one static: if the
            other one moves during this stage and comes closer than CLEARANCE, this arm waits where it is until that
            motion has ended (at most MAX_WAIT) and plans again (#141)."""
            plan = plan_fn()
            result.planning_time += plan.planning_time
            deadline = world.time + MAX_WAIT
            while plan.feasible and too_close(plan.path, duration):
                self.waits += 1
                if world.time > deadline:
                    break
                end = max(world.arms.moving_until(r) for r in others) - world._t0
                yield from wait(min(end, deadline) - world.time + 0.1)
                plan = plan_fn()
                result.planning_time += plan.planning_time
            if not plan.feasible or too_close(plan.path, duration):
                result.failed_stage = result.failed_stage or stage
                return False
            for _ in follow_path(world, plan.path, duration, robot):
                if carrying is not None and np.linalg.norm(C.getFrame(carrying).getPosition()
                                                           - gripper.getPosition()) > DROP_DISTANCE:
                    result.failed_stage = result.failed_stage or "drop"
                    return False
                yield True
            return True

        def landed(box: str) -> str:
            """Where a box that was let go of has come to rest: "belt", "region" (any), "box" (on top of another
            box), "table" or "floor" (seen with friction 0.05: the squeeze shot the box off the table)."""
            position = C.getFrame(box).getPosition()
            if box in world.conveyor.on_belt():
                return "belt"
            if position[2] < 0.3:                                             # the table top is at 0.6 m
                return "floor"
            if position[2] > world.conveyor.geometry.z_top + BOX_SIZE:      # higher than a box on the belt or table
                return "box"
            return "region" if any(region_contains(C, r, position) for r in REGIONS) else "table"

        gripper = C.getFrame(f"{ROBOT_PREFIX[robot]}gripper")

        def wait(duration: float) -> Iterator[bool]:
            end = world.time + duration
            while world.time < end - 1e-6:
                yield True

        def let_go(box: str | None) -> Iterator[bool]:
            world.arms.set_fingers(robot, FINGER_OPEN, SQUEEZE_TIME)
            if box is not None:
                world.conveyor.unhold(box)
            yield from wait(SQUEEZE_TIME + SQUEEZE_SETTLE)

        def leave(box: str | None) -> Iterator[bool]:
            """Retreat 10 cm up and go home. The box just let go of is no obstacle for the retreat: the open
            fingers may still touch it, and a plan that starts in contact is infeasible (the arm would be stuck)."""
            def retreat():
                with planning_view(world, robot), ignoring(C, box):
                    return plan_lift(C, robot, height=0.1)

            def home():
                with planning_view(world, robot):
                    return plan_home(C, robot)

            if not (yield from run("retreat", retreat, RETREAT_TIME)):
                return False
            return (yield from run("home", home, HOME_TIME))

        region = assignment.target_region
        if not self.pick_only and free_slot(C, region, world.conveyor.colors) is None:
            result.failed_stage = "full"                       # reported, not overfilled; the arm has not moved
            return
        world.arms.finger_target[robot] = FINGER_OPEN         # the pick path opens the fingers on the way
        self._picking = True                                   # observe() may re-plan the approach (#104)
        while True:
            # a re-plan starts near the box, with the fingers already around it: the planner, which takes the box as a
            # static obstacle where it is now, would find the path that follows it in collision (seen: infeasible)
            target = self._box_near(world, track) if self._replans else None
            with planning_view(world, robot), ignoring(C, target):
                pick = self._plan_pick(world, assignment, track)
            result.planning_time += pick.planning_time
            if not pick.feasible:
                result.failed_stage = "pick"
                if self._replans:                              # the arm has moved: back home, tried again later
                    self._picking = False
                    yield from leave(None)
                return
            if too_close(pick.path, assignment.pick_time - world.time):   # the timing is fixed by the moving box:
                self.waits += 1                                             # no waiting, tried again later (#141)
                result.failed_stage = "pick"
                if self._replans:
                    self._picking = False
                    yield from leave(None)
                return
            update = None
            for _ in follow_path(world, pick.path, assignment.pick_time - world.time, robot):
                update, self._update = self._update, None
                if update is not None:
                    break
                yield True
            if update is None:
                break                                          # at the box
            if update == "lost":                               # the box left the belt (pushed off): give up
                self._picking = False
                result.failed_stage = "lost"
                yield from leave(None)
                return
            track, self._replans = update, self._replans + 1   # displaced: re-plan from where the arm is now
        self._picking = False
        box = box_at_gripper(world, robot)
        if box is not None:
            world.conveyor.hold(box)
        world.arms.set_fingers(robot, FINGER_SQUEEZE, SQUEEZE_TIME)
        yield from wait(SQUEEZE_TIME + SQUEEZE_SETTLE)
        if box is None or not holding(world, robot):
            result.failed_stage = "grasp"
            yield from let_go(box)
            yield from leave(box)
            return
        result.box = box

        def lift():
            with planning_view(world, robot, box):
                return plan_lift(C, robot)

        gripper_z, box_z = gripper.getPosition()[2], C.getFrame(box).getPosition()[2]
        lifted = yield from run("lift", lift, LIFT_TIME)
        if lifted:
            yield from wait(SQUEEZE_SETTLE)
            risen = C.getFrame(box).getPosition()[2] - box_z >= gripper.getPosition()[2] - gripper_z - SLIP_TOLERANCE
            if not (risen and holding(world, robot)):
                result.failed_stage = "slip"
        if result.failed_stage is None and self.pick_only:
            result.success = True
            return
        placed, slot = False, None
        if result.failed_stage is None:
            slots = free_slots(C, region, [b for b in world.conveyor.colors if b != box])
            if not slots:
                result.failed_stage = "full"
            else:
                def place():
                    nonlocal slot
                    with planning_view(world, robot, box):
                        for slot in slots:   # the first free slot whose place can be planned (Robot 2 into the bin:
                            plan = plan_place(C, robot, box, region, clearance=self.drop, offset=SLOTS[slot])
                            if plan.feasible:   # some slots next to boxes already there could not be planned)
                                break
                    return plan

                placed = yield from run("place", place, PLACE_TIME, carrying=box)
        # release: the box is on its own now (after a failure: back on the belt or wherever it fell)
        yield from let_go(box)
        left = yield from leave(box)
        if result.failed_stage in ("slip", "drop"):
            result.dropped_on = landed(box)
        if not (left and placed):
            return
        frame = C.getFrame(box)
        result.place_error = float(np.linalg.norm(frame.getPosition()[:2] - slot_position(C, region, slot)))
        result.tilt = float(np.degrees(np.arccos(np.clip(frame.getRotationMatrix()[2, 2], -1.0, 1.0))))
        result.success = region_contains(C, region, frame.getPosition()) and result.tilt < MAX_TILT
        if not result.success:
            result.failed_stage = "release"


def pick_and_place(world: World, assignment: Assignment, track: Track,
                   on_step: Callable[[World], None] | None = None) -> PickPlaceResult:
    """Blocking form of `PickPlaceTask`: runs the world until the assignment has ended (`on_step` after every step)."""
    task = PickPlaceTask(world, assignment, track)
    while task.advance():
        world.step()
        if on_step:
            on_step(world)
    return task.result
