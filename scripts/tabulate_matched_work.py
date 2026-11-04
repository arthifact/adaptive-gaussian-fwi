"""Tabulate the paired low-frequency work experiment from the archived numerical records."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def tabulate(results):
    rows = []
    expected = {p.parent.name for p in (results / "runs").glob("*/adaptive.npz")}
    seen = set()
    for file in sorted((results / "work-matched").glob("*.npz")):
        with np.load(file, allow_pickle=False) as matched:
            settings = json.loads(matched["settings"].item())
            records = json.loads(matched["progress"].item())
            model, seed = settings["model"], settings["noise_seed"]
            case = f"{model}-seed{seed}"
            if case not in expected or case in seen or settings["method"] != "fixed_gaussians_matched_macro_work":
                raise ValueError(f"Unexpected or duplicate work-matched case in {file}")
            seen.add(case)
            reference = results / "runs" / case / "adaptive.npz"
            if hashlib.sha256(reference.read_bytes()).hexdigest() != settings["adaptive_archive_sha256"]:
                raise ValueError(f"Mismatched adaptive reference for {file}")
            with np.load(reference, allow_pickle=False) as adaptive:
                targets = json.loads(adaptive["progress"].item())[:2]
            for stage_map, record, target in zip(matched["stage_maps"], records, targets, strict=True):
                cap = target["forward_shots"] + target["adjoint_shots"]
                used = record["forward_shots"] + record["adjoint_shots"]
                if (record["frequency"] != target["frequency"] or cap != record["target_shot_count"]
                        or not 0 <= cap-used < 2*settings["survey"]["n_shots"]):
                    raise ValueError(f"Unmatched band or work budget in {file}")
                rmse = float(np.sqrt(np.mean((stage_map.astype(np.float64) - matched["true"])**2)))
                if not np.isclose(rmse, record["rmse_m_s"], rtol=1e-8):
                    raise ValueError(f"Incorrect saved model error in {file}")
                rows.append(dict(model=model, seed=seed, band_hz=record["frequency"],
                                 iterations=record["iterations"], fitted_values=record["fitted_values"],
                                 rmse_m_s=rmse, heldout_clean_loss=record["heldout_clean_loss"],
                                 adaptive_fit_forward_shots=target["forward_shots"],
                                 adaptive_fit_adjoint_shots=target["adjoint_shots"],
                                 fixed_fit_forward_shots=record["forward_shots"],
                                 fixed_fit_adjoint_shots=record["adjoint_shots"]))
    if not expected or seen != expected:
        raise ValueError(f"Expected one matched control for every adaptive reference in {results}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory containing runs/ and work-matched/")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = tabulate(args.results)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
