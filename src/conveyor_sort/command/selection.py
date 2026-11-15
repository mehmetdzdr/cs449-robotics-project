"""Command mode: choose the box and the robot that carry out a user `Command`."""
import numpy as np

from conveyor_sort.coordination.allocation import LEAD_TIME, MAX_TRACK_GAP, MIN_TRACK_AGE, PICK_X, REACH_MARGIN
from conveyor_sort.interfaces import Assignment, Command, Track
from conveyor_sort.planning.reach import is_reachable
from conveyor_sort.sim.regions import REGION_ROBOTS
from conveyor_sort.sim.scene import ROBOT_PREFIX
from conveyor_sort.sim.world import World

MIN_TIME_LEFT = 2.0   # after the pick time the box must stay in the robot's part of the belt this long [s]


def select(world: World, command: Command, tracks: list[Track], first_seen: dict[int, float],
           robots: tuple[int, ...] = (1, 2), blocked: dict[int, float] | None = None) -> Assignment | None:
    """The (box, robot) pair that carries out the command, or None if there is no feasible box (yet).

    The robot must be able to reach the target region; the box must have the commanded color, be tracked
    reliably, be reachable at the pick time (`LEAD_TIME` from now) and have at least `MIN_TIME_LEFT` in the
    robot's part of the belt after that. Of all such pairs the one with the box closest to the robot's base
    is chosen: the easiest reach. `blocked` maps track ids to the time before which they must not be chosen.
    """
    pick_time = world.time + LEAD_TIME
    candidates = []
    for robot_id in robots:
        if robot_id not in REGION_ROBOTS[command.target_region]:
            continue
        base = world.C.getFrame(f"{ROBOT_PREFIX[robot_id]}panda_base").getPosition()
        x_min, x_max = PICK_X[robot_id]
        for track in tracks:
            position = track.predict(pick_time)
            if (track.color == command.color and world.time >= (blocked or {}).get(track.track_id, 0.0)
                    and world.time - first_seen[track.track_id] >= MIN_TRACK_AGE
                    and world.time - track.timestamp <= MAX_TRACK_GAP and x_min <= position[0]
                    and (x_max - position[0]) / max(track.velocity[0], 1e-6) >= MIN_TIME_LEFT):
                candidates.append((float(np.linalg.norm(position[:2] - base[:2])), robot_id, track, position))

    for _, robot_id, track, position in sorted(candidates, key=lambda candidate: candidate[0]):
        if is_reachable(world.C, robot_id, position) and is_reachable(world.C, robot_id, position + REACH_MARGIN):
            return Assignment(robot_id, track.track_id, command.target_region, pick_time, position)
    return None
