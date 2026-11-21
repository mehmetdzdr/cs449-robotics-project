"""Covered boxes (#177): the cover, the plans of both robots, the cover decisions, cover detection, and the default
without covers. Planning only (kinematic World); the physical runs are in test_covers_physics.py."""
import inspect

import numpy as np
import pytest

from conveyor_sort.coordination.allocation import FIXED_ROLES, LEAD_TIME, allocate
from conveyor_sort.coordination.clear_and_sort import plan_clearing_job
from conveyor_sort.coordination.covers import HANDOVER_TIME, LIFTER, SORTER, Uncovering
from conveyor_sort.interfaces import BoxColor, Track
from conveyor_sort.perception.camera import Camera
from conveyor_sort.perception.covers import HsvCoverDetector, get_cover_detector
from conveyor_sort.perception.detectors import get_detector
from conveyor_sort.perception.hsv import HsvDetector
from conveyor_sort.planning.pick import own_arm_only, plan_lift, plan_moving_pick, plan_place
from conveyor_sort.planning.reach import LIFT_HEIGHT, can_pick
from conveyor_sort.sim.conveyor import BOX_SIZE, Conveyor, ConveyorConfig
from conveyor_sort.sim.covers import (COVER_INSIDE, COVER_REST, COVER_SPOT, CoverConfig, CoveredConveyor,
                                      cover_frames)
from conveyor_sort.sim.regions import DEFAULT_TARGET
from conveyor_sort.sim.scene import ROBOT_PREFIX, load_scene
from conveyor_sort.sim.world import World
from conveyor_sort.sorting import run_robots

SPEED = 0.05   # belt speed of the planning cases [m/s] (the physics tests and the demo run at it too)


def covered_world(x: float, y: float = 0.0, color: BoxColor = BoxColor.RED, speed: float = 0.0):
    world = World(ConveyorConfig(speed=speed, max_boxes=0), covers=CoverConfig())
    box = world.conveyor.spawn_box(color, x=x, y=y)
    return world, box, world.conveyor.cover(box)


def move(world: World, box: str, cover: str, dx: float) -> None:
    """Move a cover and the box under it along the belt (as the physical belt does), for planning at a later time."""
    frame = world.C.getFrame(cover)
    frame.setPosition(frame.getPosition() + [dx, 0, 0])
    box_frame = world.C.getFrame(box)
    box_frame.setRelativePosition(box_frame.getRelativePosition() + [dx, 0, 0])


# --- the default: no covers, no change --------------------------------------------------------------------------

def test_default_world_has_no_cover_frames():
    world = World(ConveyorConfig())
    assert type(world.conveyor) is Conveyor
    assert not [n for n in world.C.getFrameNames() if n.startswith("cover")]
    assert world.C.getFrameNames() == load_scene().getFrameNames()        # nothing added to the scene
    assert inspect.signature(run_robots).parameters["covers"].default is None
    assert inspect.signature(World).parameters["covers"].default is None


def decisions(covers: CoverConfig | None):
    """A stream with blocked pairs after 12 s: the boxes, the ground-truth detections, and the allocation and
    clearing decisions on tracks of them."""
    world = World(ConveyorConfig(speed=0.1, spacing=0.3, seed=3, blocked_pairs=0.3), covers=covers)
    world.run(12.0)
    boxes = {n: (world.conveyor.colors[n], tuple(np.round(world.C.getFrame(n).getPosition(), 9)))
             for n in world.conveyor.colors}
    detections = [(d.color, tuple(np.round(d.position, 9))) for d in get_detector("gt", world)()]
    tracks = [Track(i, world.conveyor.colors[n], world.C.getFrame(n).getPosition(), np.array([0.1, 0, 0]), world.time)
              for i, n in enumerate(sorted(world.conveyor.on_belt()))]
    first_seen = {t.track_id: 0.0 for t in tracks}
    assignments = [(a.robot_id, a.track_id, a.target_region, tuple(np.round(a.pick_position, 9)))
                   for a in allocate(world, tracks, first_seen, {}, FIXED_ROLES, [1, 2])]
    decision = plan_clearing_job(world, tracks, first_seen, {}, FIXED_ROLES, 6)
    job = decision.job and [(s.robot_id, s.track_id, s.region, s.across_belt) for s in decision.job.steps]
    return boxes, detections, assignments, job, decision.later, decision.infeasible


def test_the_mode_without_covers_makes_the_same_decisions():
    """With the covered boxes mode on but no cover on the belt (share 0): the same boxes, detections, allocation and
    clearing decisions as the default world."""
    default = decisions(None)
    assert default[2] or default[3]                    # the case decides something
    assert decisions(CoverConfig()) == default


def test_covers_do_not_change_the_box_stream():
    """The covers are drawn from their own random stream: the same boxes in the same places, some of them covered."""
    plain = World(ConveyorConfig(speed=0.1, spacing=0.3, seed=1))
    covered = World(ConveyorConfig(speed=0.1, spacing=0.3, seed=1), covers=CoverConfig(share=1.0))
    plain.run(25.0)
    covered.run(25.0)
    assert plain.conveyor.colors == covered.conveyor.colors
    for box in plain.conveyor.colors:
        assert np.allclose(plain.C.getFrame(box).getPosition(), covered.C.getFrame(box).getPosition(), atol=1e-9)
    under = covered.conveyor.covered()
    assert under and all(covered.conveyor.colors[b] != BoxColor.GRAY for b in under)
    assert set(covered.conveyor.on_belt()) == set(plain.conveyor.on_belt()) - set(under)   # hidden while covered


# --- the cover ----------------------------------------------------------------------------------------------------

def test_cover_encloses_the_box_and_hides_it_until_lifted():
    world, box, cover = covered_world(-0.5)
    C, conveyor = world.C, world.conveyor
    assert C.getFrame(cover).getSize() == pytest.approx(C.getFrame(box).getSize())      # the knob: grasped like a box
    assert conveyor.covered() == {box: cover} and box not in conveyor.on_belt()
    assert conveyor.covers_on_belt() == [cover]
    knob = C.getFrame(cover).getPosition()
    assert knob[:2] == pytest.approx(C.getFrame(box).getPosition()[:2])
    assert knob[2] == pytest.approx(conveyor.geometry.z_top + COVER_REST)
    C.computeCollisions()                              # 1.5 cm around the box, 2 cm above it: no contact
    parts = set(cover_frames(C, cover))
    assert not [(a, b) for a, b, _ in C.getCollisions(0.01) if {a, b} & parts and box in (a, b)]
    assert COVER_INSIDE - BOX_SIZE / 2 == pytest.approx(0.015)
    C.getFrame(cover).setPosition(knob + [0, 0, LIFT_HEIGHT])   # lifted: the box is on the belt again
    assert conveyor.covered() == {} and box in conveyor.on_belt() and conveyor.covers_on_belt() == []


def test_a_cover_and_its_box_move_together():
    world, box, cover = covered_world(-0.8, speed=0.1)
    start = {n: world.C.getFrame(n).getPosition() for n in (box, cover)}
    world.run(2.0)
    for name in (box, cover):
        assert world.C.getFrame(name).getPosition() - start[name] == pytest.approx([0.2, 0, 0], abs=1e-6)
    assert world.conveyor.covered() == {box: cover}


def test_only_a_box_with_room_around_it_is_covered():
    world = World(ConveyorConfig(max_boxes=0), covers=CoverConfig(share=1.0))
    conveyor = world.conveyor
    edge = conveyor.spawn_box(BoxColor.RED, x=-0.8, y=0.12)          # the cover would overhang the belt
    pair = [conveyor.spawn_box(BoxColor.GRAY, x=-0.3, y=0.0), conveyor.spawn_box(BoxColor.BLUE, x=-0.38, y=0.0)]
    free = conveyor.spawn_box(BoxColor.GREEN, x=0.2, y=0.05)
    gray = conveyor.spawn_box(BoxColor.GRAY, x=0.6, y=0.0)            # gray boxes are never covered
    assert set(conveyor.covered()) == {free}
    assert edge not in conveyor.covered() and pair[1] not in conveyor.covered() and gray not in conveyor.covered()


def test_spot_is_cleared_and_covers_are_reused():
    world, box, cover = covered_world(-0.5)
    conveyor = world.conveyor
    spot = world.C.getFrame(COVER_SPOT).getPosition()
    world.C.getFrame(cover).setPosition([spot[0], spot[1], spot[2] + COVER_REST])     # put aside
    assert conveyor.clear_spot() == 1 and conveyor.covers_cleared == 1
    assert world.C.getFrame(cover).getPosition()[1] < -3                               # parked out of view
    for i in range(4):                                                                 # the pool: 4 covers
        conveyor.cover(conveyor.spawn_box(BoxColor.RED, x=-0.9 + 0.3 * i, y=0.0))
    with pytest.raises(RuntimeError, match="raise CoverConfig.pool"):
        conveyor.cover(conveyor.spawn_box(BoxColor.RED, x=0.5, y=0.0))


# --- plans: Robot 2 lifts the cover and puts it aside, Robot 1 picks and places the box -------------------------

@pytest.mark.parametrize("pick_x, lateral", [(-0.55, 0.0), (-0.45, 0.05), (-0.35, -0.05), (-0.25, 0.0)])
def test_cover_then_box_plans_for_both_robots(pick_x, lateral):
    """Robot 2: moving pick of the knob (3 s, cover 15 cm upstream now), lift with the cover attached (the walls
    clear the box), put aside on the spot. Then Robot 1: moving pick of the uncovered box where it is HANDOVER_TIME
    later, place into its region."""
    world, box, cover = covered_world(pick_x - SPEED * LEAD_TIME, y=lateral)
    C = world.C
    knob = C.getFrame(cover).getPosition()
    track = Track(0, BoxColor.GRAY, knob, np.array([SPEED, 0, 0]), 0.0)
    assert can_pick(C, LIFTER, track.predict(LEAD_TIME))
    with own_arm_only(C):                                   # as on physics: the other arm stands still
        pick = plan_moving_pick(C, LIFTER, track, 0.0, LEAD_TIME)
        assert pick.feasible, "cover pick"
        move(world, box, cover, SPEED * LEAD_TIME)          # the cover has arrived where the pick ends
        C.setJointState(pick.path[-1])
        gripper = C.getFrame(f"{ROBOT_PREFIX[LIFTER]}gripper")
        assert np.linalg.norm(gripper.getPosition() - C.getFrame(cover).getPosition()) < 0.01
        frame = C.getFrame(cover)
        frame.setParent(gripper, True)                      # held (planning_view attaches it the same way)
        lift = plan_lift(C, LIFTER)
        assert lift.feasible, "cover lift"
        C.setJointState(lift.path[-1])
        box_top = C.getFrame(box).getPosition()[2] + BOX_SIZE / 2
        assert frame.getPosition()[2] - COVER_REST > box_top + 0.05          # the walls are well above the box
        place = plan_place(C, LIFTER, cover, COVER_SPOT, clearance=COVER_REST - BOX_SIZE / 2)
        assert place.feasible, "cover put aside"
        C.setJointState(place.path[-1])
        spot = C.getFrame(COVER_SPOT)
        assert np.linalg.norm(frame.getPosition()[:2] - spot.getPosition()[:2]) < 0.01
        assert frame.getPosition()[2] == pytest.approx(spot.getPosition()[2] + spot.getSize()[2] / 2 + COVER_REST, abs=0.005)
        frame.unLink()
    C.setJointState(np.concatenate([pick.path[0][:7], place.path[-1][7:14]]))   # Robot 2 stays by the spot

    handover = HANDOVER_TIME - LEAD_TIME                    # Robot 1 decides when the box track is usable
    box_track = Track(1, BoxColor.RED, C.getFrame(box).getPosition() + [SPEED * handover, 0, 0],
                      np.array([SPEED, 0, 0]), 0.0)
    C.getFrame(box).setRelativePosition(C.getFrame(box).getRelativePosition() + [SPEED * handover, 0, 0])
    assert can_pick(C, SORTER, box_track.predict(LEAD_TIME))
    with own_arm_only(C):
        pick = plan_moving_pick(C, SORTER, box_track, 0.0, LEAD_TIME)
        assert pick.feasible, "box pick"
        C.getFrame(box).setRelativePosition(C.getFrame(box).getRelativePosition() + [SPEED * LEAD_TIME, 0, 0])
        C.setJointState(pick.path[-1])
        gripper = C.getFrame(f"{ROBOT_PREFIX[SORTER]}gripper")
        C.getFrame(box).setParent(gripper, True)
        lift = plan_lift(C, SORTER)
        assert lift.feasible, "box lift"
        C.setJointState(lift.path[-1])
        assert plan_place(C, SORTER, box, DEFAULT_TARGET[BoxColor.RED]).feasible, "box place"


# --- decisions ----------------------------------------------------------------------------------------------------

def uncovering(cover_tracks: list[Track], time: float = 5.0) -> Uncovering:
    world = World(ConveyorConfig(speed=SPEED, max_boxes=0), covers=CoverConfig())
    world.time = time
    u = Uncovering(world, lambda: [])
    u.tracks = cover_tracks
    u.first_seen = {t.track_id: 0.0 for t in cover_tracks}
    return u


def cover_track(track_id: int, x: float, speed: float = SPEED, time: float = 5.0) -> Track:
    return Track(track_id, BoxColor.GRAY, np.array([x, 0.2, 0.62 + COVER_REST]), np.array([speed, 0, 0]), time)


def test_a_reachable_cover_is_lifted_now():
    u = uncovering([cover_track(0, -0.55)])                 # knob picked at -0.4, the box picked at about -0.07
    kind, track = u.decide()
    assert kind == "start" and track.track_id == 0


def test_a_cover_coming_within_reach_is_waited_for():
    u = uncovering([cover_track(0, -1.05, speed=0.1)])      # knob at -0.75 at pick time, at -0.25 within 5 s
    assert u.decide()[0] == "wait"
    free, assignments = u.start([1, 2], {}, False, [], {})
    assert free == [] and assignments == []                 # nothing else starts: the belt is kept for the cover


def test_a_cover_too_far_downstream_is_reported_and_left_alone():
    u = uncovering([cover_track(0, 0.1)])                   # the box would be past Robot 1's part of the belt
    assert u.decide() == ("none", None)
    assert u.stats.infeasible == 1 and u.blocked[0] == float("inf")
    assert u.start([1, 2], {}, False, [], {}) == ([1, 2], [])


def test_a_young_cover_track_is_not_used():
    u = uncovering([cover_track(0, -0.55)], time=0.5)       # tracked for 0.5 s (MIN_TRACK_AGE 1 s)
    assert u.decide() == ("none", None)


def test_the_uncovered_box_goes_to_robot_1():
    u = uncovering([])
    lifted_at = u.world.time - 2.0
    u._handover = (np.array([-0.4, 0.2, 0.74]), lifted_at)
    box = Track(7, BoxColor.BLUE, np.array([-0.4 + SPEED * 2.0, 0.2, 0.65]), np.array([SPEED, 0, 0]), u.world.time)
    gray = Track(8, BoxColor.GRAY, np.array([0.5, 0.2, 0.65]), np.array([SPEED, 0, 0]), u.world.time)
    free, assignments = u.start([1, 2], {}, False, [box, gray], {7: lifted_at + 0.5, 8: 0.0})
    assert [(a.robot_id, a.track_id, a.target_region) for a in assignments] == [(1, 7, "region_C")]
    assert free == [2] and u.stats.handed_over == 1 and u._handover is None


def test_robot_1_waits_for_the_uncovered_box_then_is_released():
    u = uncovering([])
    lifted_at = u.world.time - 0.5
    u._handover = (np.array([-0.4, 0.2, 0.74]), lifted_at)
    assert u.start([1, 2], {}, False, [], {}) == ([2], [])           # not seen yet: Robot 1 is kept for it
    u.world.time += 10.0                                              # never seen: Robot 1 is free again
    assert u.start([1, 2], {}, False, [], {}) == ([1, 2], []) and u.stats.lost == 1


# --- detection ----------------------------------------------------------------------------------------------------

def test_hsv_finds_the_cover_and_no_box_under_it():
    """Rendered top-camera image: the cover is found at its knob; the box detector sees the uncovered boxes, not the
    box under the cover, and takes the cover for no box."""
    world, red, cover = covered_world(-0.3, y=0.03)
    gray = world.conveyor.spawn_box(BoxColor.GRAY, x=0.1, y=-0.05)
    blue = world.conveyor.spawn_box(BoxColor.BLUE, x=0.4, y=0.05)
    obs = Camera(world.C).capture(timestamp=1.0)
    covers = HsvCoverDetector(world.conveyor.geometry).detect(obs)
    boxes = HsvDetector(world.conveyor.geometry).detect(obs)
    knob = world.C.getFrame(cover).getPosition()
    print("covers", [np.round(d.position, 4) for d in covers], "boxes", [(d.color.value, np.round(d.position, 3)) for d in boxes])
    assert len(covers) == 1 and np.linalg.norm(covers[0].position - knob) < 0.01 and covers[0].timestamp == 1.0
    assert sorted(d.color.value for d in boxes) == ["blue", "gray"]
    for d, name in zip(sorted(boxes, key=lambda d: d.color.value), (blue, gray)):
        assert np.linalg.norm(d.position - world.C.getFrame(name).getPosition()) < 0.01
    assert get_cover_detector("gt", world)()[0].position == pytest.approx(knob)
    assert [d.color for d in get_detector("gt", world)()] == [BoxColor.GRAY, BoxColor.BLUE]   # red: covered
    world.C.getFrame(cover).setPosition(knob + [0, 0, 0.3])           # lifted: no cover on the belt, red seen
    assert get_cover_detector("gt", world)() == [] and BoxColor.RED in [d.color for d in get_detector("gt", world)()]


def test_covered_mode_world_is_a_covered_conveyor():
    world = World(ConveyorConfig(max_boxes=0), covers=CoverConfig(pool=2))
    assert isinstance(world.conveyor, CoveredConveyor) and world.conveyor.covers == ["cover_0", "cover_1"]
    assert COVER_SPOT in world.C.getFrameNames()
