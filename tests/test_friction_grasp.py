"""#87: a static box is lifted by finger friction only, in the BotOp physics simulation."""
import pytest

from conveyor_sort.physics.friction_grasp import HOLDING_MIN, run_lift_trial


@pytest.mark.parametrize("kwargs", [
    {"mass": 0.05}, {"mass": 0.2}, {"yaw_deg": 30}, {"robot_id": 2},
], ids=["0.05 kg", "0.2 kg", "rotated 30 deg", "Robot 2"])
def test_box_is_lifted_and_held_by_friction(kwargs):
    trial = run_lift_trial(**kwargs)
    assert trial.holding
    assert not trial.attached                              # no kinematic shortcut: the box stays a free body
    assert trial.gripper_lift > 0.12                       # the arm went up (some tracking error under gravity)
    assert trial.lift == pytest.approx(trial.gripper_lift, abs=0.005)   # and the box went with it: no slip
    assert trial.tilt < 5.0
    assert abs(trial.grasp_offset) < 0.005                 # grasped at the box centre


def test_box_slips_out_with_low_friction():
    """Proof that friction is what holds the box: with friction 0.05 the same grasp loses it."""
    trial = run_lift_trial(friction=0.05)
    assert not trial.attached
    assert trial.lift < 0.02


def test_closing_on_nothing_is_detected():
    trial = run_lift_trial(with_box=False)
    assert not trial.holding
    assert trial.finger_q < HOLDING_MIN
