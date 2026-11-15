"""Plot the Gaussian macro model and grid refinement of a main.py run."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]


def plot_runs(paths):
    """Save result.png next to each adaptive run (folder or result.npz)."""
    for path in paths:
        file = path / "result.npz" if path.is_dir() else path
        with np.load(file, allow_pickle=False) as data:
            settings = json.loads(str(data["settings"]))
            if settings["method"] not in (
                "adaptive_gaussian_macro",
                "gaussian_background_pixels",
            ):
                raise ValueError(f"Expected an adaptive Gaussian run: {file}")
            truth = data["true"]
            panels = [("Simulation truth", truth), ("Starting ramp", data["initial"])]
            panels.extend(
                (f"Gaussian macro model\n{band:g} Hz", grid)
                for band, grid in zip(
                    data["gaussian_frequencies"],
                    data["gaussian_backgrounds"],
                    strict=True,
                )
            )
            if "pixel_models" in data:
                bands = (
                    data["pixel_frequencies"]
                    if "pixel_frequencies" in data
                    else settings["pixel_frequencies"]
                )
                panels.extend(
                    (f"Grid refinement\n{band:g} Hz", grid)
                    for band, grid in zip(bands, data["pixel_models"], strict=True)
                )
            if any(
                grid.shape != truth.shape for _, grid in panels
            ) or not np.array_equal(panels[-1][1], data["final"]):
                raise ValueError(f"Inconsistent stage maps in {file}")

        nz, nx = truth.shape
        dx = settings["survey"]["dx"]
        low, high = settings["survey"]["bounds"]
        extent = (-dx / 2, (nx - 0.5) * dx, (nz - 0.5) * dx, -dx / 2)
        fig = Figure(figsize=(3.1 * len(panels), 3.7), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, len(panels))
        for ax, (label, grid) in zip(axes, panels, strict=True):
            image = ax.imshow(
                grid,
                interpolation="nearest",
                extent=extent,
                cmap="viridis",
                vmin=low,
                vmax=high,
            )
            ax.set(title=label, xlabel="Distance (m)")
        axes[0].set_ylabel("Depth (m)")
        fig.colorbar(image, ax=list(axes), label="Velocity (m/s)", shrink=0.85)
        output = file.with_name("result.png")
        fig.savefig(output, dpi=200)
        print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        type=Path,
        nargs="*",
        help="adaptive run folders or result.npz files (default: all under results/)",
    )
    args = parser.parse_args()
    paths = args.runs or sorted((ROOT / "results").glob("*/adaptive*/result.npz"))
    if paths:
        plot_runs(paths)
    else:
        print("No adaptive runs in results/ yet.")
