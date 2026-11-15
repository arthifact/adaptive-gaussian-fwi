"""Tabulate the ablations: no pruning, and the growth threshold halved or doubled.

Each ablation is a macro-only run (6 and 9 Hz), e.g. results/<model>-seed<seed>/
adaptive_no_pruning_macro/, compared with the 6 and 9 Hz stages of adaptive/ on
the same data. Writes paper/tables/ablations.csv and a scores.csv per ablation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from paper.score_comparison import ssim

ROOT = Path(__file__).resolve().parents[1]
PAIRED_FIELDS = (
    "model",
    "noise_seed",
    "observation_sha256",
    "input_sha256",
    "versions",
)


def read(file):
    """Settings, 6 and 9 Hz progress and 9 Hz SSIM of an adaptive run."""
    with np.load(file, allow_pickle=False) as data:
        settings = json.loads(data["settings"].item())
        progress = json.loads(data["progress"].item())
        if [p["frequency"] for p in progress[:2]] != [6, 9]:
            raise ValueError(f"Expected 6 and 9 Hz Gaussian stages in {file}")
        similarity = ssim(
            data["gaussian_backgrounds"][1],
            data["true"],
            float(np.ptp(settings["survey"]["bounds"])),
        )
    return settings, progress[:2], similarity


def row(settings, stages, similarity, variant):
    """One table row: the 9 Hz macro model of one run."""
    return dict(
        model=settings["model"],
        seed=settings["noise_seed"],
        variant=variant,
        pruning=settings["gaussian_pruning"],
        threshold_scale=settings.get("threshold_scale", 1.0),
        gaussians_9_hz=stages[1]["gaussians"],
        fitted_values_9_hz=6 * stages[1]["gaussians"],
        rmse_6_hz_m_s=stages[0]["rmse"],
        rmse_9_hz_m_s=stages[1]["rmse"],
        ssim_9_hz=similarity,
        forward_shots_through_9_hz=sum(s["forward_shots"] for s in stages),
        adjoint_shots_through_9_hz=sum(s["adjoint_shots"] for s in stages),
    )


def tabulate(root=ROOT):
    """Rows for each case with ablations (standard run first) and their folders."""
    rows, folders = [], []
    for reference in sorted((root / "results").glob("*/adaptive/result.npz")):
        ablations = sorted(reference.parent.parent.glob("adaptive_*_macro/result.npz"))
        if not ablations:
            continue
        settings, stages, similarity = read(reference)
        rows.append(row(settings, stages, similarity, "standard"))
        folders.append(None)
        for file in ablations:
            other, other_stages, other_similarity = read(file)
            if any(other[key] != settings[key] for key in PAIRED_FIELDS):
                raise ValueError(
                    f"{file} does not use the same observations as {reference}"
                )
            rows.append(
                row(
                    other,
                    other_stages,
                    other_similarity,
                    file.parent.name.removeprefix("adaptive_").removesuffix("_macro"),
                )
            )
            folders.append(file.parent)
    if not rows:
        raise ValueError(f"No ablation runs found under {root / 'results'}")
    return rows, folders


def write_csv(path, rows):
    """Write `rows` (dicts with the same keys) to a CSV file."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper" / "tables" / "ablations.csv",
        help="combined CSV",
    )
    args = parser.parse_args()
    rows, folders = tabulate()
    write_csv(args.output, rows)
    for record, folder in zip(rows, folders, strict=True):
        if folder is not None:
            write_csv(folder / "scores.csv", [record])
