"""Repeat the MAE stage in the exact Python runtime used by the JEPA runs."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_cost_benchmark import measure


if __name__ == "__main__":
    measure("mae-vits-torch2141", ["/home/coder/venv/bin/python", "tools/run_mae_baseline.py",
                                    "--config", "configs/continue10_corruption.json",
                                    "--run", "runs/cost-mae-vits-seed17-torch2141"])
