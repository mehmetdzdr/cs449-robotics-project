"""Planning interface for the rest of the system: carry out one Assignment (pick a moving box, place it, go home)."""
from dataclasses import dataclass
from typing import Callable

from conveyor_sort.interfaces import Assignment, Track
from conveyor_sort.planning.execute import execute_path, grasp, release
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


def pick_and_place(world: World, assignment: Assignment, track: Track,
                   on_step: Callable[[World], None] | None = None) -> PickPlaceResult:
    """Plan and execute the assignment stage by stage; the world keeps running meanwhile (`on_step` after every step).

    The box is grasped at `assignment.pick_time` where `track` predicts it. Each stage is planned from the
    state the previous one ended in. On failure the arm stops where it is (after a failed grasp it goes home).
    """
    C, robot = world.C, assignment.robot_id
    result = PickPlaceResult(False, 0.0, None, None)

    def run(stage: str, plan: Plan, duration: float) -> bool:
        result.planning_time += plan.planning_time
        if not plan.feasible:
            result.failed_stage = result.failed_stage or stage
            return False
        execute_path(world, plan.path, duration, on_step)
        return True

    if not run("pick", plan_moving_pick(C, robot, track, world.time, assignment.pick_time),
               assignment.pick_time - world.time):
        return result
    result.box = grasp(world, robot)
    if result.box is None:
        result.failed_stage = "grasp"
        run("home", plan_home(C, robot), HOME_TIME)
        return result
    result.success = (
        run("lift", plan_lift(C, robot), LIFT_TIME)
        and run("place", plan_place(C, robot, result.box, assignment.target_region), PLACE_TIME)
        and (release(world, result.box, assignment.target_region) or True)
        and run("retreat", plan_lift(C, robot, height=0.1), RETREAT_TIME)
        and run("home", plan_home(C, robot), HOME_TIME)
    )
    return result
