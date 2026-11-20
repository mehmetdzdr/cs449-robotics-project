"""CI setup for the physics tests (#97): slow tests are marked, every PR runs the fast subset, a failure is retried once."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_slow_marker_is_registered(pytestconfig):
    assert any(line.startswith("slow:") for line in pytestconfig.getini("markers"))


def test_pull_requests_run_the_fast_subset_and_the_full_suite_runs_nightly():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert 'pytest -q -m "not slow" --reruns 1' in ci and "pytest -q --reruns 1" in ci
    assert "schedule:" in ci and "workflow_dispatch:" in ci


def test_slow_tests_are_marked():
    """Every test module that runs the physics scenarios marks them slow (module-level `pytestmark`)."""
    for name in ("test_loops_physics.py", "test_shared_area_physics.py", "test_physical_place.py", "test_scripts.py",
                 "test_friction_grasp.py"):
        assert "pytestmark = pytest.mark.slow" in (ROOT / "tests" / name).read_text(), name
    assert "@pytest.mark.slow\n@pytest.mark.parametrize(\"robot_id, region, pick_x\"" in (ROOT / "tests" / "test_region_slots.py").read_text()
