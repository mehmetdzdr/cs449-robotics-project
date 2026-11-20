"""Buffer (#101, #149): each robot can take a box from every buffer slot to its own region. Planning only (kinematic
World, KOMO), as PickPlaceTask does it: stationary pick, lift, place into the first slot of the region; if that place
cannot be planned, carry the box home first and place from there.

Measured in CI: the direct place was infeasible from 4 of 6 slots for Robot 2 (into the bin) and 1 of 6 for Robot 1
(the arm swings about 180 deg around its base); carried home first, Robot 2's place worked from every slot, Robot 1's
into region A from none. Hence the carry home only as a fallback.
"""
import numpy as np

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.planning.pick import plan_home, plan_lift, plan_pick, plan_place
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.regions import BUFFER, SLOTS, slot_position
from conveyor_sort.sim.scene import ROBOT_PREFIX
from conveyor_sort.sim.world import World

TARGET = {1: "region_A", 2: "bin_gray"}


def take_from_buffer(robot_id: int, slot: int) -> str:
    """The path that works ("direct" or "via home"), or a description of the failure."""
    world = World(ConveyorConfig(max_boxes=0))
    C, conveyor = world.C, world.conveyor
    box = conveyor.spawn_box(BoxColor.GRAY)
    conveyor.attach(box, BUFFER)
    C.getFrame(box).setPosition([*slot_position(C, BUFFER, slot), 0.633])

    def place(steps=20):
        return plan_place(C, robot_id, box, TARGET[robot_id], steps=steps, offset=SLOTS[0])

    pick = plan_pick(C, robot_id, box)
    if not pick.feasible:
        return "pick infeasible"
    C.setJointState(pick.path[-1])
    conveyor.attach(box, f"{ROBOT_PREFIX[robot_id]}gripper")
    lift = plan_lift(C, robot_id)
    if not lift.feasible:
        return "lift infeasible"
    C.setJointState(lift.path[-1])
    if place().feasible:
        return "direct"
    home = plan_home(C, robot_id)
    if not home.feasible:
        return "direct place and carry home infeasible"
    C.setJointState(home.path[-1])
    if place().feasible:
        return "via home"
    C.computeCollisions()
    near = [(a, b, round(d, 3)) for a, b, d in C.getCollisions(0.01)]
    return (f"place infeasible both ways; from home: near-collisions {near}, gripper at "
            f"{np.round(C.getFrame(f'{ROBOT_PREFIX[robot_id]}gripper').getPosition(), 2)}, "
            f"40 steps feasible: {place(40).feasible}")


def test_every_buffer_slot_can_be_taken_to_the_region():
    table = {(r, s): take_from_buffer(r, s) for r in (1, 2) for s in range(len(SLOTS))}
    report = "\n".join(f"robot {r} slot {s} {np.round(SLOTS[s], 2)}: {path}" for (r, s), path in table.items())
    print(report)
    assert all(path in ("direct", "via home") for path in table.values()), report
