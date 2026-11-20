"""Command mode end to end: text -> Command -> box and robot -> pick and place in the running simulation."""
from dataclasses import dataclass
from typing import Callable

from conveyor_sort.command.parser import parse_command
from conveyor_sort.command.selection import select
from conveyor_sort.interfaces import Command, Detection, Track
from conveyor_sort.planning.pick_place import PickPlaceTask
from conveyor_sort.prediction.tracker import Tracker
from conveyor_sort.sim.regions import region_contains
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import RETRY_DELAY, collect_if_full


@dataclass
class CommandResult:
    command: Command
    success: bool               # a box was placed in the target region
    correct: bool               # ground truth: that box has the commanded color and lies inside the region
    robot_id: int | None        # robot that carried it out (None: no feasible box before the timeout)
    box: str | None             # frame name of the box
    failed_stage: str | None    # "timeout", or the pick-and-place stage that failed
    time: float                 # simulation time from the command to its end [s]


class CommandMode:
    """Executes user commands one after the other while the belt keeps running.

    `detect()` returns the detections of the current camera frame. Boxes that nobody asks for travel on
    and leave the belt.
    """

    def __init__(self, world: World, detect: Callable[[], list[Detection]],
                 on_step: Callable[[World], None] | None = None):
        self.world, self.detect, self.on_step = world, detect, on_step
        self.tracker = Tracker()
        self.tracks: list[Track] = []
        self.first_seen: dict[int, float] = {}
        self.last_placed: dict[str, str] = {}      # region -> box lying in it

    def _step(self) -> None:
        self.world.step()
        self.tracks = self.tracker.update(self.detect(), self.world.time)
        for track in self.tracks:
            self.first_seen.setdefault(track.track_id, self.world.time)
        if self.on_step:
            self.on_step(self.world)

    def execute(self, command: Command | str, timeout: float = 60.0) -> CommandResult:
        """Wait for a feasible box of the commanded color (at most `timeout` s), then pick and place it.

        A text command is parsed first (`ValueError` if it is invalid).
        """
        world = self.world
        if isinstance(command, str):
            command = parse_command(command)
        start = world.time
        blocked: dict[int, float] = {}
        while world.time - start < timeout:
            assignment = select(world, command, self.tracks, self.first_seen, blocked=blocked)
            if assignment is None:
                self._step()
                continue
            track = next(t for t in self.tracks if t.track_id == assignment.track_id)
            collect_if_full(world, command.target_region)   # physics: a full region is emptied first (#142)
            task = PickPlaceTask(world, assignment, track)
            if not task.advance():                 # the pick cannot be planned from here: try again a little later
                blocked[track.track_id] = world.time + RETRY_DELAY
                continue
            if command.target_region in self.last_placed:   # a region holds one box: the previous one is collected
                world.conveyor.remove(self.last_placed.pop(command.target_region))
            self._step()
            while task.advance():
                self._step()
                task.observe(self.tracks)      # physics: a box displaced during the approach is aimed at again (#104)
            result = task.result
            if result.success:
                if world.bot is None:          # physics: regions hold several boxes (slots, #93)
                    self.last_placed[command.target_region] = result.box
            correct = (result.success and world.conveyor.colors[result.box] == command.color
                       and region_contains(world.C, command.target_region, world.C.getFrame(result.box).getPosition()))
            return CommandResult(command, result.success, correct, assignment.robot_id, result.box,
                                 result.failed_stage, world.time - start)
        return CommandResult(command, False, False, None, None, "timeout", world.time - start)
