"""Physical pick on the BotOp backend: squeeze a box between the fingers, check the grasp, plan while holding."""
from contextlib import contextmanager, nullcontext

import numpy as np

from conveyor_sort.physics.friction_grasp import FINGER_OPEN, HOLDING_MIN
from conveyor_sort.sim.scene import ROBOT_PREFIX

GRASP_RADIUS = 0.03   # a box centre this close to the gripper centre at pick time is the one to squeeze [m]
SQUEEZE_TIME = 0.4    # finger motion from open to the squeeze reference [s]
SQUEEZE_SETTLE = 0.4  # then wait before the finger position is read [s]
PENETRATION = 0.003   # planning_view leaves out boxes that penetrate a non-robot object deeper than this [m]
SLIP_TOLERANCE = 0.02  # the box may rise this much less than the gripper during the lift before it counts as a slip [m]


def box_at_gripper(world, robot_id: int, boxes: list[str] | None = None) -> str | None:
    """The box on the belt (or of `boxes`) whose centre is closest to the gripper centre, if within GRASP_RADIUS."""
    gripper = world.C.getFrame(f"{ROBOT_PREFIX[robot_id]}gripper").getPosition()
    boxes = world.conveyor.on_belt() if boxes is None else boxes
    distances = {b: np.linalg.norm(world.C.getFrame(b).getPosition() - gripper) for b in boxes}
    box = min(distances, key=distances.get, default=None)
    return box if box is not None and distances[box] < GRASP_RADIUS else None


def holding(world, robot_id: int) -> bool:
    """The fingers stopped on something after the squeeze (see friction_grasp.HOLDING_MIN)."""
    return float(world.C.getJointState()[world.fingers[robot_id]]) > HOLDING_MIN


@contextmanager
def ignoring(C, box: str | None):
    """For planning only: `box` is no collision object (its contact flag in C; the simulation keeps its shapes)."""
    frame = C.getFrame(box) if box else None
    if frame is not None:
        frame.setContact(0)
    try:
        yield
    finally:
        if frame is not None:
            frame.setContact(1)


@contextmanager
def planning_view(world, robot_id: int, box: str | None = None):
    """What KOMO sees on the physics backend: every robot's fingers open, and `box` (squeezed by `robot_id`'s
    gripper, if given) attached to that gripper.

    Squeezed fingers penetrate their box slightly, which the planner reports as a collision: of the planning
    robot, and also of the other robot (static for the planner), whose penetration made every plan of this robot
    infeasible while the other held a box. Likewise a box that penetrates another box, the belt or the table by
    more than PENETRATION is left out (with `box`: the other one), as the planner cannot resolve that contact and
    would refuse every plan: e.g. the squeeze centres the grasped box along the belt and pushes it into the box
    jammed behind it. A free box would not move with the gripper in the planner. Only C is
    changed, and it is restored afterwards (the simulation never sees it: the next sync overwrites the box pose
    and the joints anyway).

    While the other arm stands still, only `robot_id`'s arm is planned (`planning.pick.own_arm_only`, #141).
    """
    from conveyor_sort.planning.pick import own_arm_only
    C = world.C
    q = C.getJointState()
    view = q.copy()
    view[list(world.fingers.values())] = FINGER_OPEN
    C.setJointState(view)
    C.computeCollisions()
    stuck = {f for a, b, _ in C.getCollisions(-PENETRATION) if not _robot_frame(a) and not _robot_frame(b)
             for f in (a, b) if f in world.conveyor.colors and f != box}   # with `box`: the other one is left out
    for name in stuck:
        C.getFrame(name).setContact(0)
    frame = C.getFrame(box) if box else None
    if frame is not None:
        frame.setParent(C.getFrame(f"{ROBOT_PREFIX[robot_id]}gripper"), True)
    now = world.arms.bot.get_t()
    static = all(world.arms.moving_until(r) <= now for r in world.fingers if r != robot_id)
    try:
        with own_arm_only(C) if static else nullcontext():
            yield
    finally:
        if frame is not None:
            frame.unLink()
        for name in stuck:
            C.getFrame(name).setContact(1)
        C.setJointState(q)


def _robot_frame(name: str) -> bool:
    return name.startswith(tuple(ROBOT_PREFIX.values()))
