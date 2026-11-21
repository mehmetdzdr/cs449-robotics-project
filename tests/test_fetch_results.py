"""Reproducible results (#112): where the files of the results release go (no network access in this test)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_results  # noqa: E402


def test_weights_and_results_go_where_the_code_reads_them():
    assert fetch_results.target("yolo_boxes.pt") == ROOT / "weights" / "yolo_boxes.pt"     # perception.yolo default
    assert fetch_results.target("clearing.csv") == ROOT / "results" / "clearing.csv"        # make_figures --results
    assert fetch_results.target("SOURCES.md") == ROOT / "results" / "SOURCES.md"


def test_the_release_is_the_one_the_readme_links():
    assert f"releases/tag/{fetch_results.RELEASE}" in (ROOT / "README.md").read_text(encoding="utf-8")
