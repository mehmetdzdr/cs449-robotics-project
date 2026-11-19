"""Joint references of both arms, merged into the one spline that BotOp follows.

BotOp drives all joints with a single spline (`bot.move`). Each robot here has its own reference: timed
waypoints on BotOp's clock. When one robot gets a new path, the references of both robots are sampled at all
their future waypoint times and sent as one spline that overwrites BotOp's current one from now on
(`overwrite=True`). The other robot's waypoints are kept, so its motion is not disturbed.

With active finger joints (`friction_grasp.activate_fingers`) each robot's reference also has its finger joint.
The finger of a path is not taken from the path (KOMO keeps it at the planning value) but held at the robot's
finger target, which only `set_fingers` changes: a squeeze stays on while the arm moves.
"""
import numpy as np
import robotic as ry

from conveyor_sort.sim.scene import ROBOT_PREFIX, arm_joints

MIN_DURATION = 0.1   # a path is never squeezed into less than this [s]
KNOT_SPACING = 0.05  # the merged spline has a knot at least this often, so BotOp's spline stays close to the
                     # piecewise-linear references (fewer knots: the other arm deviated more when one is re-planned)
LATENCY = 0.1        # the arm follows BotOp's spline 0.1 s late (measured: joint error 74 to 82 mrad mean against
                     # the reference during a pick, 4 to 10 mrad against the reference shifted by 0.1 s), so every
                     # reference is sent this much earlier: the arm itself then arrives at the planned time


class ArmReferences:
    def __init__(self, bot: ry.BotOp, C: ry.Config, fingers: dict[int, int] | None = None, finger_target: float = 0.04):
        """`fingers`: robot_id -> index of its finger joint in q, if the fingers are active joints."""
        self.bot = bot
        self.fingers = fingers or {}
        self.joints = {r: arm_joints(C, r) + ([self.fingers[r]] if r in self.fingers else []) for r in ROBOT_PREFIX}
        self.finger_target = {r: finger_target for r in self.fingers}
        now, q = bot.get_t(), bot.get_q()
        self.nq = len(q)
        # robot_id -> (times on BotOp's clock, waypoints of the robot's joints); a single waypoint means "hold"
        self.plans = {r: (np.array([now]), q[None, j].copy()) for r, j in self.joints.items()}

    def _reference(self, robot_id: int, t: float) -> np.ndarray:
        times, waypoints = self.plans[robot_id]
        return np.array([np.interp(t, times, waypoints[:, j]) for j in range(waypoints.shape[1])])

    def reference(self, robot_id: int, t: float) -> np.ndarray:
        """Arm joint reference of one robot at BotOp time t (linear between waypoints, constant outside); the arm
        is expected there at t + LATENCY."""
        return self._reference(robot_id, t)[:7]

    def set_path(self, robot_id: int | None, path: np.ndarray, end: float) -> None:
        """Robot `robot_id` (None: both) follows `path` (KOMO path, all joints; only this robot's are used) from
        now, reaching its last waypoint at BotOp time `end`; the waypoints are evenly spaced in time."""
        now = self.bot.get_t()
        end = max(end - LATENCY, now + MIN_DURATION)
        times = now + (end - now) * np.arange(1, len(path) + 1) / len(path)
        for r, joints in self.joints.items():
            if robot_id is None or r == robot_id:
                waypoints = path[:, joints].copy()
                if r in self.fingers:
                    waypoints[:, -1] = self.finger_target[r]
                self._set(r, now, times, waypoints)
        self._send(now)

    def set_fingers(self, robot_id: int, target: float, duration: float) -> None:
        """Move one robot's finger to `target` in `duration` seconds; its arm holds its current reference."""
        now = self.bot.get_t()
        self.finger_target[robot_id] = target
        waypoint = self._reference(robot_id, now)
        waypoint[-1] = target
        self._set(robot_id, now, np.array([now + max(duration, MIN_DURATION)]), waypoint[None])
        self._send(now)

    def _set(self, robot_id: int, now: float, times: np.ndarray, waypoints: np.ndarray) -> None:
        self.plans[robot_id] = (np.concatenate([[now], times]), np.vstack([self._reference(robot_id, now), waypoints]))

    def _send(self, now: float) -> None:
        knots = np.concatenate([t[t > now + 1e-3] for t, _ in self.plans.values()])
        if len(knots) == 0:
            return
        knots = np.unique(np.concatenate([knots, np.arange(now + KNOT_SPACING, knots.max(), KNOT_SPACING)]))
        spline = np.zeros((len(knots), self.nq))
        for r, joints in self.joints.items():
            spline[:, joints] = [self._reference(r, t) for t in knots]
        # Known limitation: the overwrite briefly disturbs the other arm (max joint error against its reference
        # 28 mrad while the other arm is re-planned, 8 to 18 mrad without); starting it later made it worse
        self.bot.move(spline, knots - now, overwrite=True, overwriteCtrlTime=now)
