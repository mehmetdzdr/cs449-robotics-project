"""Buffer (#101, #149): each robot can take a box from every buffer slot to its own region. Planning only (kinematic
World, KOMO): stationary pick, lift, place into the first slot of the region."""
import numpy as np

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.planning.pick import plan_lift, plan_pick, plan_place
from conveyor_sort.sim.conveyor import ConveyorConfig
from conveyor_sort.sim.regions import BUFFER, SLOTS, slot_position
from conveyor_sort.sim.scene import ROBOT_PREFIX
from conveyor_sort.sim.world import World

TARGET = {1: "region_A", 2: "bin_gray"}


def stages(robot_id: int, slot: int) -> dict[str, bool]:
    world = World(ConveyorConfig(max_boxes=0))
    C, conveyor = world.C, world.conveyor
    box = conveyor.spawn_box(BoxColor.GRAY)
    conveyor.attach(box, BUFFER)
    C.getFrame(box).setPosition([*slot_position(C, BUFFER, slot), 0.633])
    feasible = {}
    for stage, plan_fn in [("pick", lambda: plan_pick(C, robot_id, box)),
                           ("lift", lambda: plan_lift(C, robot_id)),
                           ("place", lambda: plan_place(C, robot_id, box, TARGET[robot_id], offset=SLOTS[0]))]:
        plan = plan_fn()
        feasible[stage] = plan.feasible
        if not plan.feasible:
            break
        C.setJointState(plan.path[-1])
        if stage == "pick":
            conveyor.attach(box, f"{ROBOT_PREFIX[robot_id]}gripper")
    return feasible


def test_every_buffer_slot_can_be_taken_to_the_region():
    table = {(r, s): stages(r, s) for r in (1, 2) for s in range(len(SLOTS))}
    report = "\n".join(f"robot {r} slot {s} {np.round(SLOTS[s], 2)}: {f}" for (r, s), f in table.items())
    print(report)
    assert all(len(f) == 3 and all(f.values()) for f in table.values()), report
