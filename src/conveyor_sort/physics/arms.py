"""Joint references of both arms, merged into the one spline that BotOp follows.

BotOp drives all joints with a single spline (`bot.move`). Each robot here has its own reference: timed
waypoints on BotOp's clock. When one robot gets a new path, the references of both robots are sampled at all
their future waypoint times and sent as one spline that overwrites BotOp's current one from now on
(`overwrite=True`). The other robot's waypoints are kept, so its motion is not disturbed.
"""
import numpy as np
import robotic as ry

ARM_JOINTS = {1: slice(0, 7), 2: slice(7, 14)}   # robot_id -> its joints in q (see sim.scene.ROBOT_PREFIX)
MIN_DURATION = 0.1   # a path is never squeezed into less than this [s]


class ArmReferences:
    def __init__(self, bot: ry.BotOp):
        self.bot = bot
        now, q = bot.get_t(), bot.get_q()
        # robot_id -> (times on BotOp's clock, waypoints); a single waypoint means "hold"
        self.plans = {r: (np.array([now]), q[None, joints].copy()) for r, joints in ARM_JOINTS.items()}

    def reference(self, robot_id: int, t: float) -> np.ndarray:
        """Joint reference of one robot at BotOp time t (linear between waypoints, constant outside)."""
        times, waypoints = self.plans[robot_id]
        return np.array([np.interp(t, times, waypoints[:, j]) for j in range(waypoints.shape[1])])

    def end_time(self, robot_id: int) -> float:
        return float(self.plans[robot_id][0][-1])

    def set_path(self, robot_id: int | None, path: np.ndarray, end: float) -> None:
        """Robot `robot_id` (None: both) follows `path` (KOMO path, all joints; only this robot's are used) from
        now, reaching its last waypoint at BotOp time `end`; the waypoints are evenly spaced in time."""
        now = self.bot.get_t()
        end = max(end, now + MIN_DURATION)
        times = now + (end - now) * np.arange(1, len(path) + 1) / len(path)
        for r, joints in ARM_JOINTS.items():
            if robot_id is None or r == robot_id:
                self.plans[r] = (np.concatenate([[now], times]), np.vstack([self.reference(r, now), path[:, joints]]))
        knots = np.unique(np.concatenate([t[t > now + 1e-3] for t, _ in self.plans.values()]))
        spline = np.array([np.concatenate([self.reference(r, t) for r in ARM_JOINTS]) for t in knots])
        self.bot.move(spline, knots - now, overwrite=True, overwriteCtrlTime=now)
