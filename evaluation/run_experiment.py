"""Run experiments from config files and log their metrics.

    python evaluation/run_experiment.py evaluation/configs/example.json
    python evaluation/run_experiment.py evaluation/configs/*.json --out results/runs.csv     (headless: xvfb-run -a ...)

Each run prints its record as JSON and appends it as one row to the CSV file.
"""
import argparse
import json

from conveyor_sort.metrics import append_csv, load_config, run_experiment

parser = argparse.ArgumentParser()
parser.add_argument("configs", nargs="+", help="experiment config files (JSON)")
parser.add_argument("--out", default="results/runs.csv", help="CSV file the records are appended to")
args = parser.parse_args()

for path in args.configs:
    record = run_experiment(load_config(path))
    print(json.dumps(record, indent=2))
    append_csv(args.out, record)
print(f"appended {len(args.configs)} record(s) to {args.out}")
