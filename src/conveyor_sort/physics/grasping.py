"""Physical pick on the BotOp backend: squeeze a box between the fingers, check the grasp, plan while holding."""
from contextlib import contextmanager

import numpy as np

from conveyor_sort.physics.friction_grasp import FINGER_OPEN, HOLDING_MIN
from conveyor_sort.sim.scene import ROBOT_PREFIX

GRASP_RADIUS = 0.03   # a box centre this close to the gripper centre at pick time is the one to squeeze [m]
SQUEEZE_TIME = 0.4    # finger motion from open to the squeeze reference [s]
SQUEEZE_SETTLE = 0.4  # then wait before the finger position is read [s]
SLIP_TOLERANCE = 0.02  # the box may rise this much less than the gripper during the lift before it counts as a slip [m]


def box_at_gripper(world, robot_id: int) -> str | None:
    """The box on the belt whose centre is closest to the gripper centre, if within GRASP_RADIUS."""
    gripper = world.C.getFrame(f"{ROBOT_PREFIX[robot_id]}gripper").getPosition()
    distances = {b: np.linalg.norm(world.C.getFrame(b).getPosition() - gripper) for b in world.conveyor.on_belt()}
    box = min(distances, key=distances.get, default=None)
    return box if box is not None and distances[box] < GRASP_RADIUS else None


def holding(world, robot_id: int) -> bool:
    """The fingers stopped on something after the squeeze (see friction_grasp.HOLDING_MIN)."""
    return float(world.C.getJointState()[world.fingers[robot_id]]) > HOLDING_MIN


@contextmanager
def planning_view(world, robot_id: int, box: str | None):
    """What KOMO sees while the gripper squeezes `box`: open fingers and the box attached to the gripper.

    The squeezed fingers penetrate the box slightly, which the planner would report as a collision; and a free
    box would not move with the gripper in the planner. Only C is changed, and it is restored afterwards (the
    simulation never sees it: the next sync overwrites the box pose and the joints anyway).
    """
    C = world.C
    q = C.getJointState()
    view = q.copy()
    view[world.fingers[robot_id]] = FINGER_OPEN
    C.setJointState(view)
    frame = C.getFrame(box) if box else None
    if frame is not None:
        frame.setParent(C.getFrame(f"{ROBOT_PREFIX[robot_id]}gripper"), True)
    try:
        yield
    finally:
        if frame is not None:
            frame.unLink()
        C.setJointState(q)
