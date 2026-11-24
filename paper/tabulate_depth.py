"""Tabulate the 9 Hz macro-model error of both Gaussian methods by depth.

The five shots light the shallow part of each model much better than the deep
part, so the rows are split into a top, a middle and a bottom third.
"""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = dict(
    adaptive="results/{case}/adaptive",
    fixed_gaussians="comparisons/results/{case}/fixed_gaussians",
)


def depth_rmse(model, true):
    """Velocity RMSE of the top, middle and bottom third of the rows."""
    error = np.square(model - true)
    return [float(np.sqrt(part.mean())) for part in np.array_split(error, 3)]


def tabulate(root=ROOT):
    """One row per case and method, plus the shared starting model."""
    rows = []
    for reference in sorted((root / "results").glob("*/adaptive/result.npz")):
        case = reference.parent.parent.name
        for method, folder in RUNS.items():
            with np.load(root / folder.format(case=case) / "result.npz") as data:
                settings = json.loads(data["settings"].item())
                recorded = json.loads(data["progress"].item())[1]["rmse"]
                true, start, macro = data["true"], data["initial"], data["macro"]
            # The archive's own 9 Hz error must be the error of this macro model.
            assert np.isclose(np.sqrt(np.square(macro - true).mean()), recorded)
            for name, model in (("start", start), (method, macro)):
                if name == "start" and method != "adaptive":
                    continue
                top, middle, bottom = depth_rmse(model, true)
                rows.append(
                    dict(
                        model=settings["model"],
                        seed=settings["noise_seed"],
                        method=name,
                        rmse_top_m_s=top,
                        rmse_middle_m_s=middle,
                        rmse_bottom_m_s=bottom,
                    )
                )
    if not rows:
        raise ValueError(f"No adaptive runs found under {root / 'results'}")
    return rows


if __name__ == "__main__":
    rows = tabulate()
    output = ROOT / "paper" / "tables" / "depth.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")
