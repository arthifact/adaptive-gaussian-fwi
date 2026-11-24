"""Tabulate the fixed lattice's 6 and 9 Hz error for every Adam step that was run.

The reported lattice uses the step chosen on Marmousi seed 1 (MACRO_RATE in
comparisons/fixed_gaussians.py). The other steps show what a different choice
would give on each case.
"""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def tabulate(root=ROOT):
    """One row per fixed-lattice run, excluding the work-matched runs."""
    rows = []
    for folder in sorted((root / "comparisons" / "results").glob("*/fixed_gaussians*")):
        archive = folder / "result.npz"
        if "_matched" in folder.name or not archive.is_file():
            continue
        with np.load(archive, allow_pickle=False) as data:
            settings = json.loads(data["settings"].item())
            progress = json.loads(data["progress"].item())
        rows.append(
            dict(
                model=settings["model"],
                seed=settings["noise_seed"],
                adam_step_m_s=settings["macro_adam_rate"],
                rmse_6_hz_m_s=progress[0]["rmse"],
                rmse_9_hz_m_s=progress[1]["rmse"],
            )
        )
    if not rows:
        raise ValueError("No fixed-lattice runs found")
    return sorted(rows, key=lambda r: (r["model"], r["seed"], r["adam_step_m_s"]))


if __name__ == "__main__":
    rows = tabulate()
    output = ROOT / "paper" / "tables" / "lattice-steps.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")
