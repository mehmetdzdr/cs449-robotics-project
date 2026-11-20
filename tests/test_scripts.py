"""The demo entry points run from the command line."""
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

pytestmark = pytest.mark.slow   # physics scenarios in real time: nightly / on demand in CI (#97)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(script: str, *args: str) -> str:
    result = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


@pytest.mark.parametrize("mode, sorted_boxes", [("single", 1), ("fixed", 1), ("coordinated", 1)])
def test_run_autonomous(mode, sorted_boxes):
    out = run("run_autonomous.py", "--mode", mode, "--duration", "14", "--speed", "0.1", "--seed", "0", "--detector", "gt")
    assert f"sorted correctly: {sorted_boxes}, failed: 0" in out
    assert ("shared area" in out) == (mode != "single")


def test_run_autonomous_clear_and_sort_scene_and_video(tmp_path):
    out = run("run_autonomous.py", "--scene", "clear-and-sort")
    # the demos run on physics (#96), where clear-and-sort is off: the blocked pair is left alone, no arm contact
    assert "sorted correctly: 0, failed: 0, left the belt: 2" in out and "0 contacts" in out
    video = tmp_path / "demo.mp4"
    assert f"saved {video}" in run("run_autonomous.py", "--mode", "single", "--duration", "3", "--video", str(video))
    assert video.stat().st_size > 10_000
    capture = cv2.VideoCapture(str(video))           # plays in real time: physics steps are longer than dt (the
    length = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)   # frame render alone takes
    assert length == pytest.approx(3.0, abs=0.3)     # 0.13 s), one frame per step gave a 3x too fast video


def test_run_command():
    out = run("run_command.py", "pick a blue box and place it at B", "pick a box", "--speed", "0.1", "--seed", "0")
    assert "'pick a blue box and place it at B': done, robot 1" in out
    assert "rejected: no box color in 'pick a box'" in out


def test_invalid_option_is_rejected():
    result = subprocess.run([sys.executable, str(SCRIPTS / "run_autonomous.py"), "--detector", "magic"], capture_output=True, text=True)
    assert result.returncode != 0 and "invalid choice" in result.stderr
