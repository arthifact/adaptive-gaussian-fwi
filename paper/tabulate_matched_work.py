"""Tabulate the matched-work fixed-lattice runs.

Every case needs the run at the reported Adam step; runs at other steps
(fixed_gaussians_matched_rate<step>/) are added when present. Writes the
combined table and a scores.csv in each run folder.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from comparisons.fixed_gaussians import MACRO_RATE
from paper.score_comparison import ssim

ROOT = Path(__file__).resolve().parents[1]


def tabulate(root=ROOT):
    """Rows of the matched-work table, and the run folder of each row."""
    rows, folders = [], []
    results = root / "results"
    expected = {p.parent.parent.name for p in results.glob("*/adaptive/result.npz")}
    seen = set()
    for file in sorted(
        (root / "comparisons" / "results").glob("*/fixed_gaussians_matched*/result.npz")
    ):
        with np.load(file, allow_pickle=False) as matched:
            settings = json.loads(matched["settings"].item())
            records = json.loads(matched["progress"].item())
            model, seed = settings["model"], settings["noise_seed"]
            case = f"{model}-seed{seed}"
            rate = settings["macro_adam_rate"]
            if (
                case not in expected
                or (case, rate) in seen
                or settings["method"] != "fixed_gaussians_matched_macro_work"
            ):
                raise ValueError(f"Unexpected or duplicate work-matched case in {file}")
            seen.add((case, rate))
            reference = results / case / "adaptive" / "result.npz"
            if (
                hashlib.sha256(reference.read_bytes()).hexdigest()
                != settings["adaptive_archive_sha256"]
            ):
                raise ValueError(f"Mismatched adaptive reference for {file}")
            with np.load(reference, allow_pickle=False) as adaptive:
                targets = json.loads(adaptive["progress"].item())[:2]
            for stage_map, record, target in zip(
                matched["stage_maps"], records, targets, strict=True
            ):
                cap = target["forward_shots"] + target["adjoint_shots"]
                used = record["forward_shots"] + record["adjoint_shots"]
                if (
                    record["frequency"] != target["frequency"]
                    or cap != record["target_shot_count"]
                    or not 0 <= cap - used < 2 * settings["survey"]["n_shots"]
                ):
                    raise ValueError(f"Unmatched band or work budget in {file}")
                rmse = float(
                    np.sqrt(
                        np.mean((stage_map.astype(np.float64) - matched["true"]) ** 2)
                    )
                )
                if not np.isclose(rmse, record["rmse_m_s"], rtol=1e-8):
                    raise ValueError(f"Incorrect saved model error in {file}")
                rows.append(
                    dict(
                        model=model,
                        seed=seed,
                        adam_step_m_s=rate,
                        band_hz=record["frequency"],
                        iterations=record["iterations"],
                        fitted_values=record["fitted_values"],
                        rmse_m_s=rmse,
                        ssim=ssim(
                            stage_map,
                            matched["true"],
                            float(np.ptp(settings["survey"]["bounds"])),
                        ),
                        heldout_clean_loss=record["heldout_clean_loss"],
                        adaptive_fit_forward_shots=target["forward_shots"],
                        adaptive_fit_adjoint_shots=target["adjoint_shots"],
                        fixed_fit_forward_shots=record["forward_shots"],
                        fixed_fit_adjoint_shots=record["adjoint_shots"],
                    )
                )
                folders.append(file.parent)
    if not expected or {c for c, r in seen if r == MACRO_RATE} != expected:
        raise ValueError(
            f"Expected one matched control for every adaptive reference in {root}"
        )
    return rows, folders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper" / "tables" / "matched-work.csv",
        help="combined CSV",
    )
    args = parser.parse_args()
    rows, folders = tabulate()

    def write_csv(path, selected):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)

    write_csv(args.output, rows)
    for folder in sorted(set(folders)):
        write_csv(
            folder / "scores.csv",
            [r for r, f in zip(rows, folders, strict=True) if f == folder],
        )
