"""Spike for #87: lift a static 6 cm box by finger friction only (BotOp physics), for several cases.

    xvfb-run -a python scripts/spikes/friction_grasp.py
"""
from conveyor_sort.physics.friction_grasp import run_lift_trial

CASES = [  # (label, keyword arguments)
    ("0.1 kg", {}),
    ("0.05 kg", {"mass": 0.05}),
    ("0.2 kg", {"mass": 0.2}),
    ("1.0 kg", {"mass": 1.0}),
    ("rotated 30 deg", {"yaw_deg": 30}),
    ("Robot 2", {"robot_id": 2}),
    ("friction 0.05", {"friction": 0.05}),
    ("nothing between the fingers", {"with_box": False}),
]

print(f"{'case':30s} {'finger q':>9s} {'holding':>8s} {'box lift':>9s} {'gripper lift':>13s} {'tilt':>6s} {'grasp z':>8s} {'attached':>9s}")
for label, kwargs in CASES:
    t = run_lift_trial(**kwargs)
    print(f"{label:30s} {t.finger_q:9.4f} {str(t.holding):>8s} {t.lift:9.3f} {t.gripper_lift:13.3f} {t.tilt:6.1f} {t.grasp_offset * 1000:6.1f}mm {str(t.attached):>9s}")
