"""BotOp helpers: start the physics simulation and drive the simulated Franka grippers."""
import time
from importlib.metadata import version

import robotic as ry

MIN_ROBOTIC = (0, 4, 3)   # 0.4.2 does not launch simulated grippers
GRIPPER_START_DELAY = 0.3  # gripperDone() is briefly True right after a gripper command, before the motion starts [s]


def check_robotic_version() -> None:
    installed = tuple(int(p) for p in version("robotic").split(".")[:3])
    if installed < MIN_ROBOTIC:
        raise RuntimeError(f"robotic {version('robotic')} is installed, but simulated grippers need robotic >= "
                           f"{'.'.join(map(str, MIN_ROBOTIC))}: pip install -U 'robotic>=0.4.3'")


def start_simulation(C: ry.Config, hyper_speed: float = 1.0) -> ry.BotOp:
    """Start BotOp in simulation mode (physics engine) for the robots in C; no viewer window.

    `hyper_speed`: simulation time per wall-clock time (`botsim/hyperSpeed`); the simulation runs freely in its
    own thread at this rate. Measured real-time factors are in tests/test_world.py.
    """
    check_robotic_version()
    # always a float: params_add does not overwrite a parameter with a value of another type (an int 5 after
    # the float 1.0 was silently ignored)
    ry.params_add({"botsim/verbose": 0, "physx/verbose": 0, "botsim/hyperSpeed": float(hyper_speed)})
    return ry.BotOp(C, useRealRobot=False)


def wait_gripper(bot: ry.BotOp, C: ry.Config, gripper_id: int, timeout: float = 5.0) -> bool:
    """Block until the gripper command has finished; False on timeout."""
    bot.sync(C, GRIPPER_START_DELAY)
    start = time.time()
    while not bot.gripperDone(gripper_id):
        if time.time() - start > timeout:
            return False
        bot.sync(C, 0.05)
    return True
