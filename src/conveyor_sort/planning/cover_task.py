"""Covered boxes (#177): lift a cover off the moving belt by its knob and put it aside, as one non-blocking task.

The stages and the planners are those of `PickPlaceTask` on the physics backend (moving pick with the lift checked
first, squeeze, lift, place, open, retreat, home; the other arm checked with `ArmReferences.clearance`); only the
object differs: the knob of a cover instead of a box, and the cover spot instead of a region slot. The place is
`plan_place` with the cover's resting height (`COVER_REST`) as clearance above a box's. Physics backend only.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import robotic as ry

from conveyor_sort.interfaces import Assignment, Track
from conveyor_sort.physics.friction_grasp import FINGER_OPEN, FINGER_SQUEEZE
from conveyor_sort.physics.grasping import (SLIP_TOLERANCE, SQUEEZE_SETTLE, SQUEEZE_TIME, box_at_gripper, holding,
                                            planning_view)
from conveyor_sort.planning.execute import follow_path
from conveyor_sort.planning.pick import Plan, plan_home, plan_lift, plan_moving_pick, plan_place
from conveyor_sort.planning.pick_place import (CLEARANCE, DROP_DISTANCE, HOME_TIME, LIFT_TIME, MAX_TILT, MAX_WAIT,
                                               PLACE_TIME, RETREAT_TIME)
from conveyor_sort.sim.conveyor import BOX_SIZE
from conveyor_sort.sim.covers import COVER_REST, COVER_SPOT, cover_frames, tilt
from conveyor_sort.sim.scene import HOME_Q, ROBOT_PREFIX, arm_joints
from conveyor_sort.sim.world import World

SPOT_TOLERANCE = 0.05   # a cover put aside must stand this close to the spot centre (xy) [m]
# After letting go of a cover the arm retreats RETREAT_HEIGHT up and goes home keeping COVER_MARGIN from it. With a
# box's 10 cm retreat and plain plan_home, the way home from the spot knocked the cover over in every loop run in CI
# (tilt 108 to 114 deg, 0.2 to 0.6 m away), while the other arm moved and both arms were planned: the cover stood
# untouched after the release and after the retreat, so the path home swept through it between KOMO's waypoints.
RETREAT_HEIGHT = 0.2   # [m]
COVER_MARGIN = 0.03    # [m]


@dataclass
class CoverResult:
    success: bool = False           # the cover stands upright on the spot and the arm is home
    failed_stage: str | None = None   # "pick", "grasp", "lift", "slip", "place", "drop", "release", "retreat", "home"
    planning_time: float = 0.0      # summed KOMO solve time [s]
    cover: str | None = None        # frame name of the grasped cover
    grasped_at: float | None = None   # time the fingers closed on the knob [s]
    lift_started: float | None = None   # time the lift motion began (after any wait for the other arm): the box
    #                                     under the cover is free from about then [s]
    lifted_at: float | None = None  # time the cover was checked lifted off its box (after the lift) [s]
    lifted_from: np.ndarray | None = None   # knob position at the grasp (the box is below it) [m]
    carry_tilt: float = 0.0         # largest tilt of the cover in the gripper (lift and place) [deg]
    place_error: float | None = None  # cover to spot centre, xy [m]
    tilt: float | None = None       # cover tilt on the spot [deg]


def plan_home_clear(C: ry.Config, robot_id: int, cover: str, steps: int = 20) -> Plan:
    """`plan_home`, with the robot's hand and forearm at least COVER_MARGIN from every part of `cover`."""
    from conveyor_sort.planning.pick import _OWN_ARM_ONLY, _arm_shapes, _komo, _solve
    q = C.getJointState()
    komo = _komo(C, steps, robot_id)
    komo.addControlObjective([], 2, 1e0)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    for shape in _arm_shapes(C, robot_id):
        for part in cover_frames(C, cover):
            komo.addObjective([], ry.FS.negDistance, [shape, part], ry.OT.ineq, [1e1], [-COVER_MARGIN])
    for r, prefix in ROBOT_PREFIX.items():      # targets as in plan_home
        if r == robot_id or id(C) not in _OWN_ARM_ONLY:
            target = HOME_Q if r == robot_id else q[arm_joints(C, r)]
            komo.addObjective([1.0], ry.FS.qItself, [f"{prefix}panda_joint{i}" for i in range(1, 8)], ry.OT.eq, [1e1], target)
    komo.addObjective([1.0], ry.FS.qItself, [], ry.OT.eq, [1e1], [], 1)
    return _solve(komo)


@contextmanager
def ignoring_cover(C, cover: str | None):
    """For planning only: no part of `cover` is a collision object (like grasping.ignoring for a box)."""
    frames = [C.getFrame(n) for n in cover_frames(C, cover)] if cover else []
    for frame in frames:
        frame.setContact(0)
    try:
        yield
    finally:
        for frame in frames:
            frame.setContact(1)


class CoverTask:
    """Robot `assignment.robot_id` lifts the cover whose knob `track` follows at `assignment.pick_time` and puts it
    on the cover spot. Same use as `PickPlaceTask` (`advance()` before every `world.step()`, `at_belt` until the
    lift, `waits`), so the sorting loop runs it like one; `result` is a `CoverResult`."""

    box = None    # not a box taken from the buffer (sorting loop)

    def __init__(self, world: World, assignment: Assignment, track: Track):
        if world.bot is None:
            raise ValueError("CoverTask needs the physics backend (World(..., physics=True, covers=...))")
        self.assignment = assignment
        self.result = CoverResult()
        self.at_belt = True
        self.waits = 0
        self._steps = self._run(world, assignment, track)

    def advance(self) -> bool:
        return next(self._steps, False)

    def observe(self, tracks: list[Track]) -> None:
        """Box tracks (sorting loop): not used, the cover is followed by its own tracker."""

    def _run(self, world: World, assignment: Assignment, track: Track) -> Iterator[bool]:
        C, robot, result = world.C, assignment.robot_id, self.result
        conveyor = world.conveyor
        others = [r for r in world.fingers if r != robot]
        gripper = C.getFrame(f"{ROBOT_PREFIX[robot]}gripper")

        def too_close(path, duration: float) -> bool:
            now = world.time + world._t0
            return (any(world.arms.moving_until(r) > now for r in others)
                    and world.arms.clearance(C, robot, path, duration) < CLEARANCE)

        def run(stage: str, plan_fn, duration: float, carrying: str | None = None) -> Iterator[bool]:
            """As in PickPlaceTask: plan, wait for the other arm if it comes too close (#141), follow the plan; with
            `carrying`, stop as soon as the cover has left the gripper ("drop")."""
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
            if stage == "lift":
                result.lift_started = world.time
            for _ in follow_path(world, plan.path, duration, robot):
                if carrying is not None:
                    result.carry_tilt = max(result.carry_tilt, tilt(C, carrying))
                    if np.linalg.norm(C.getFrame(carrying).getPosition() - gripper.getPosition()) > DROP_DISTANCE:
                        result.failed_stage = result.failed_stage or "drop"
                        return False
                yield True
            return True

        def wait(duration: float) -> Iterator[bool]:
            end = world.time + duration
            while world.time < end - 1e-6:
                yield True

        def let_go(cover: str | None) -> Iterator[bool]:
            world.arms.set_fingers(robot, FINGER_OPEN, SQUEEZE_TIME)
            if cover is not None:
                conveyor.unhold(cover)
            yield from wait(SQUEEZE_TIME + SQUEEZE_SETTLE)

        def leave(cover: str | None) -> Iterator[bool]:
            def retreat():
                with planning_view(world, robot), ignoring_cover(C, cover):
                    return plan_lift(C, robot, height=RETREAT_HEIGHT)

            def home():
                with planning_view(world, robot):
                    clear = plan_home_clear(C, robot, cover) if cover is not None else None
                    return clear if clear is not None and clear.feasible else plan_home(C, robot)   # e.g. a cover
                    #                                                     dropped next to the arm: home anyway

            if not (yield from run("retreat", retreat, RETREAT_TIME)):
                return False
            return (yield from run("home", home, HOME_TIME))

        conveyor.clear_spot()                                  # the arm is not there: it is free, at home
        world.arms.finger_target[robot] = FINGER_OPEN
        with planning_view(world, robot):
            pick = plan_moving_pick(C, robot, track, world.time, assignment.pick_time)
            if pick.feasible:                                  # the lift from the grasp pose, before moving
                q = C.getJointState()
                C.setJointState(pick.path[-1])
                lift = plan_lift(C, robot)
                C.setJointState(q)
                pick = Plan(pick.path, lift.feasible, pick.planning_time + lift.planning_time)
        result.planning_time += pick.planning_time
        if not pick.feasible or too_close(pick.path, assignment.pick_time - world.time):
            self.waits += pick.feasible
            result.failed_stage = "pick"                       # the arm has not moved: tried again later
            return
        for _ in follow_path(world, pick.path, assignment.pick_time - world.time, robot):
            yield True
        cover = box_at_gripper(world, robot, conveyor.covers_on_belt())
        if cover is not None:
            conveyor.hold(cover)
            result.lifted_from = C.getFrame(cover).getPosition().copy()
        result.grasped_at = world.time
        world.arms.set_fingers(robot, FINGER_SQUEEZE, SQUEEZE_TIME)
        yield from wait(SQUEEZE_TIME + SQUEEZE_SETTLE)
        if cover is None or not holding(world, robot):
            result.failed_stage = "grasp"
            yield from let_go(cover)
            yield from leave(cover)
            return
        result.cover = cover

        def lift():
            with planning_view(world, robot, cover):
                return plan_lift(C, robot)

        gripper_z, cover_z = gripper.getPosition()[2], C.getFrame(cover).getPosition()[2]
        lifted = yield from run("lift", lift, LIFT_TIME, carrying=cover)
        self.at_belt = self.at_belt and not lifted
        if lifted:
            yield from wait(SQUEEZE_SETTLE)
            risen = C.getFrame(cover).getPosition()[2] - cover_z >= gripper.getPosition()[2] - gripper_z - SLIP_TOLERANCE
            if risen and holding(world, robot):
                result.lifted_at = world.time
            else:
                result.failed_stage = "slip"
        placed = False
        if result.failed_stage is None:
            def place():
                with planning_view(world, robot, cover):     # resting height: COVER_REST instead of half a box
                    return plan_place(C, robot, cover, COVER_SPOT, clearance=COVER_REST - BOX_SIZE / 2)

            first = [place()]                          # planned once: run() takes it, re-plans only after a wait
            if not first[0].feasible:
                # seen in CI once, for a cover lifted far downstream; it was let go above the belt and Robot 1's hand
                # touched it there. Carried home first, as a box from the buffer was (#149)
                def carry():
                    with planning_view(world, robot, cover):
                        return plan_home(C, robot)

                yield from run("carry", carry, HOME_TIME, carrying=cover)
                first.clear()                          # from home: plan again
            if result.failed_stage is None:
                placed = yield from run("place", lambda: first.pop() if first else place(), PLACE_TIME, carrying=cover)
        yield from let_go(cover)
        left = yield from leave(cover)
        if not (left and placed):
            return
        position = C.getFrame(cover).getPosition()
        result.place_error = float(np.linalg.norm(position[:2] - C.getFrame(COVER_SPOT).getPosition()[:2]))
        result.tilt = tilt(C, cover)
        result.success = result.place_error < SPOT_TOLERANCE and result.tilt < MAX_TILT
        if not result.success:
            result.failed_stage = "release"
