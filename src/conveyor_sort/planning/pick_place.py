"""Planning interface for the rest of the system: carry out one Assignment (pick a moving box, place it, go home)."""
from dataclasses import dataclass
from typing import Callable, Iterator

from conveyor_sort.interfaces import Assignment, Track
from conveyor_sort.planning.execute import follow_path, grasp, release
from conveyor_sort.planning.pick import Plan, plan_home, plan_lift, plan_moving_pick, plan_place
from conveyor_sort.sim.world import World

# durations of the stages after the grasp [s]
LIFT_TIME, PLACE_TIME, RETREAT_TIME, HOME_TIME = 1.0, 3.0, 1.0, 2.0


@dataclass
class PickPlaceResult:
    success: bool
    planning_time: float            # summed KOMO solve time of all stages [s]
    failed_stage: str | None        # "pick", "grasp", "lift", "place", "retreat", "home", or None on success
    box: str | None                 # frame name of the grasped box


class PickPlaceTask:
    """Carries out one assignment step by step, so that the caller can run both robots at the same time.

    Call `advance()` once before every `world.step()`: it moves this robot's arm (only its own joints) by one
    simulation step and returns False once the task has ended; `result` is final then. The box is grasped at
    `assignment.pick_time` where `track` predicts it. Each stage is planned from the state the previous one ended
    in, with the other arm treated as static at its current pose. On failure the arm stops where it is (after a
    failed grasp it goes home).
    """

    def __init__(self, world: World, assignment: Assignment, track: Track, across_belt: bool = False):
        self.assignment = assignment
        self.across_belt = across_belt     # grasp on the box faces at -y / +y (see plan_moving_pick)
        self.result = PickPlaceResult(False, 0.0, None, None)
        self._steps = self._run(world, assignment, track)

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

        if not (yield from run("pick", plan_moving_pick(C, robot, track, world.time, assignment.pick_time,
                                                        across_belt=self.across_belt),
                               assignment.pick_time - world.time)):
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


def pick_and_place(world: World, assignment: Assignment, track: Track,
                   on_step: Callable[[World], None] | None = None) -> PickPlaceResult:
    """Blocking form of `PickPlaceTask`: runs the world until the assignment has ended (`on_step` after every step)."""
    task = PickPlaceTask(world, assignment, track)
    while task.advance():
        world.step()
        if on_step:
            on_step(world)
    return task.result
