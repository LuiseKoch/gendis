"""Repository paths, resolved relative to the repo root.

Scripts live in subdirectories but always read and write the same shared
locations, so every path is anchored to the repo root rather than to the
current working directory.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
PROCESSED = DATA / "processed.csv"
PREDICTIONS = ROOT / "predictions"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for _d in (DATA, PREDICTIONS, RESULTS):
    _d.mkdir(exist_ok=True)
