"""Ground-truth detector: reads true box poses from the simulator instead of using vision.

Lets prediction, planning and coordination be developed before perception is ready, and
serves as the reference when measuring perception error.
"""
from conveyor_sort.interfaces import Detection
from conveyor_sort.sim.world import World


class GroundTruthDetector:
    def __init__(self, world: World):
        self.world = world

    def detect(self, observation=None) -> list[Detection]:
        """One Detection per box on the belt. `observation` is unused (kept for the common detector signature)."""
        conveyor = self.world.conveyor
        return [
            Detection(
                color=conveyor.colors[name],
                position=self.world.C.getFrame(name).getPosition(),
                confidence=1.0,
                timestamp=self.world.time,
            )
            for name in conveyor.on_belt()
        ]
