"""Planning interface for the rest of the system: carry out one Assignment (pick a moving box, place it, go home)."""
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from conveyor_sort.interfaces import Assignment, Track
from conveyor_sort.physics.friction_grasp import FINGER_OPEN, FINGER_SQUEEZE
from conveyor_sort.physics.grasping import (SLIP_TOLERANCE, SQUEEZE_SETTLE, SQUEEZE_TIME, box_at_gripper, holding,
                                            planning_view)
from conveyor_sort.planning.execute import follow_path, grasp, release
from conveyor_sort.planning.pick import Plan, plan_home, plan_lift, plan_moving_pick, plan_place
from conveyor_sort.sim.regions import region_contains
from conveyor_sort.sim.scene import ROBOT_PREFIX
from conveyor_sort.sim.world import World

# durations of the stages after the grasp [s]
LIFT_TIME, PLACE_TIME, RETREAT_TIME, HOME_TIME = 1.0, 3.0, 1.0, 2.0
MAX_TILT = 10.0   # physics backend: a placed box must be upright within this [deg]


@dataclass
class PickPlaceResult:
    success: bool
    planning_time: float            # summed KOMO solve time of all stages [s]
    failed_stage: str | None        # "pick", "grasp", "slip", "lift", "place", "release", "retreat", "home", or None
    box: str | None                 # frame name of the grasped box
    place_error: float | None = None   # physics backend: placed box centre to region centre in xy [m]
    tilt: float | None = None          # physics backend: placed box's tilt from upright [deg]


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
    gripper during the lift. The place is physical too: the box is lowered to its resting height in the region
    (`drop` higher, if given), the fingers open and the box is on its own while the arm retreats and goes home.
    `result.place_error` (box centre to region centre in xy) and `result.tilt` are measured at the end; failed
    stage "release": the box is not upright (tilt >= MAX_TILT) or not inside the region. `pick_only=True` ends
    after the lift (success = the box is held).
    """

    def __init__(self, world: World, assignment: Assignment, track: Track, across_belt: bool = False,
                 pick_only: bool = False, drop: float = 0.0):
        self.assignment = assignment
        self.across_belt = across_belt     # grasp on the box faces at -y / +y (see plan_moving_pick)
        self.pick_only = pick_only         # physics backend: end after the lift (success = the box is held)
        self.drop = drop                   # physics backend: release the box this much higher than resting [m]
        self.result = PickPlaceResult(False, 0.0, None, None)
        self._steps = self._run_physical(world, assignment, track) if world.bot is not None else \
            self._run(world, assignment, track)

    def advance(self) -> bool:
        return next(self._steps, False)

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

        def run(stage: str, plan: Plan, duration: float) -> Iterator[bool]:
            result.planning_time += plan.planning_time
            if not plan.feasible:
                result.failed_stage = result.failed_stage or stage
                return False
            for _ in follow_path(world, plan.path, duration, robot):
                yield True
            return True

        def wait(duration: float) -> Iterator[bool]:
            end = world.time + duration
            while world.time < end - 1e-6:
                yield True

        world.arms.finger_target[robot] = FINGER_OPEN         # the pick path opens the fingers on the way
        if not (yield from run("pick", self._plan_pick(world, assignment, track), assignment.pick_time - world.time)):
            return
        box = box_at_gripper(world, robot)
        if box is not None:
            world.conveyor.hold(box)
        world.arms.set_fingers(robot, FINGER_SQUEEZE, SQUEEZE_TIME)
        yield from wait(SQUEEZE_TIME + SQUEEZE_SETTLE)
        if box is None or not holding(world, robot):
            result.failed_stage = "grasp"
            if box is not None:
                world.conveyor.unhold(box)
            world.arms.set_fingers(robot, FINGER_OPEN, SQUEEZE_TIME)
            yield from wait(SQUEEZE_TIME)
            yield from run("home", plan_home(C, robot), HOME_TIME)
            return
        result.box = box
        with planning_view(world, robot, box):
            lift = plan_lift(C, robot)
        gripper = C.getFrame(f"{ROBOT_PREFIX[robot]}gripper")
        gripper_z, box_z = gripper.getPosition()[2], C.getFrame(box).getPosition()[2]
        if not (yield from run("lift", lift, LIFT_TIME)):
            return
        yield from wait(SQUEEZE_SETTLE)
        if C.getFrame(box).getPosition()[2] - box_z < gripper.getPosition()[2] - gripper_z - SLIP_TOLERANCE:
            result.failed_stage = "slip"
            world.conveyor.unhold(box)
            return
        if not holding(world, robot):
            result.failed_stage = "slip"
            return
        if self.pick_only:
            result.success = True
            return
        region = assignment.target_region
        with planning_view(world, robot, box):
            place = plan_place(C, robot, box, region, clearance=self.drop)
        if not (yield from run("place", place, PLACE_TIME)):
            return
        world.arms.set_fingers(robot, FINGER_OPEN, SQUEEZE_TIME)       # release: the box is on its own now
        world.conveyor.unhold(box)
        yield from wait(SQUEEZE_TIME + SQUEEZE_SETTLE)
        if not ((yield from run("retreat", plan_lift(C, robot, height=0.1), RETREAT_TIME))
                and (yield from run("home", plan_home(C, robot), HOME_TIME))):
            return
        frame = C.getFrame(box)
        result.place_error = float(np.linalg.norm(frame.getPosition()[:2] - C.getFrame(region).getPosition()[:2]))
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
