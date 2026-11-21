"""Download the YOLO weights and the raw benchmark results of a GitHub Release (#112).

    python scripts/fetch_results.py                       the release the committed figures and tables come from
    python scripts/fetch_results.py --tag results-...     another one

`yolo_boxes.pt` goes into weights/, the CSV files (and SOURCES.md, which names the script and PR behind each) into
results/. Then `python evaluation/make_figures.py` regenerates evaluation/figures/ (tables.md and the plots).
"""
import argparse
import json
import urllib.request
from pathlib import Path

REPO = "mehmetdzdr/cs449-robotics-project"
RELEASE = "results-2026-11-21"       # the release evaluation/figures/ was generated from
ROOT = Path(__file__).resolve().parents[1]


def assets(tag: str) -> list[tuple[str, str]]:
    """(name, download url) of every file attached to the release."""
    with urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/releases/tags/{tag}") as response:
        release = json.load(response)
    return [(a["name"], a["browser_download_url"]) for a in release["assets"]]


def target(name: str) -> Path:
    return ROOT / ("weights" if name.endswith(".pt") else "results") / name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=RELEASE)
    args = parser.parse_args()
    for name, url in assets(args.tag):
        path = target(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
        print(f"{path.relative_to(ROOT)}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
