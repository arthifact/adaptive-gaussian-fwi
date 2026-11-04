"""Score complete, paired default-survey runs on the true models and four unseen shots."""

import argparse
import csv
from contextlib import closing
import json
from pathlib import Path

import deepwave
import numpy as np
import torch

from src.simulation import FWI, PML_WIDTH_CELLS

METHODS = ("gaussian_background_pixels", "fixed_gaussians", "pixels_plain", "pixels_smoothed")
BANDS = (6, 9, 13, 20, 30)
BUDGETS = (325, 325, 350, 350, 350)
HELD_OUT_SHOTS = (8, 26, 43, 61)


def read_run(path):
    path = path / "result.npz" if path.is_dir() else path
    with np.load(path, allow_pickle=False) as data:
        settings = json.loads(data["settings"].item())
        progress = json.loads(data["progress"].item())
        if settings["method"] not in METHODS:
            raise ValueError(f"Unexpected method in {path}")
        if (tuple(p["frequency"] for p in progress) != BANDS
                or tuple(len(p["loss"]) for p in progress) != BUDGETS):
            raise ValueError(f"Expected the complete five-band default budget in {path}")
        maps = (np.concatenate([data["gaussian_backgrounds"], data["pixel_models"]])
                if "gaussian_backgrounds" in data else data["stage_maps"])
        truth, initial = data["true"], data["initial"]
        if (maps.shape != (5, *truth.shape) or initial.shape != truth.shape
                or not np.isfinite(maps).all() or not np.isfinite(truth).all() or not np.isfinite(initial).all()
                or not np.array_equal(data["macro"], maps[1]) or not np.array_equal(data["final"], maps[4])):
            raise ValueError(f"Inconsistent stage maps in {path}")
        macro_parameters = truth.size
        if settings["method"] == METHODS[0]:
            macro_parameters = data["gaussian_params_1"].size
        elif settings["method"] == METHODS[1]:
            macro_parameters = data["amplitudes_9"].size
        return settings, progress, truth, initial, maps[[1, 3, 4]], macro_parameters


def survey_settings(settings, truth):
    survey = dict(settings["survey"])
    survey["shape"] = tuple(survey["shape"])
    if survey["n_receivers"] is None:
        survey["n_receivers"] = survey["shape"][1]
    survey["surface_velocity"] = float(truth[0].max())
    if survey["shape"] != (70, 70) or survey["n_shots"] != 5:
        raise ValueError("The published held-out positions require the 70 × 70 five-shot survey")
    return survey


def refined_traces(survey, grid, accuracy):
    """Double grid sampling while preserving physical extent, acquisition and PML thickness."""
    velocity = torch.from_numpy(grid)
    fine_shape = tuple(2*n - 1 for n in grid.shape)
    fine = torch.nn.functional.interpolate(velocity[None, None], size=fine_shape, mode="bilinear",
                                           align_corners=True)[0, 0].contiguous()
    shots = []
    for source in survey.sources:
        waves = deepwave.scalar(fine, survey.dx / 2, survey.dt,
                                source_amplitudes=survey.wavelet.to(fine),
                                source_locations=2*source[None], receiver_locations=2*survey.receivers,
                                accuracy=accuracy, pml_width=2*PML_WIDTH_CELLS,
                                pml_freq=survey.source_frequency, max_vel=survey.bounds[1])
        shots.append(waves[-1][0])
    return torch.stack(shots)


def band_loss(problem, predicted, observed, band):
    residual = problem.lowpass(predicted - observed, band)
    return float(residual.square().sum() / problem.lowpass(observed, band).square().sum())


def compare_pair(runs):
    if set(runs) != set(METHODS):
        raise ValueError("Each model and seed needs exactly one run from each of the four methods")
    settings, _, truth, initial, _, _ = runs[METHODS[0]]
    survey = survey_settings(settings, truth)
    pair_fields = ("model", "noise_seed", "noise_db", "input_sha256", "observation_sha256", "versions")
    for method, (other, _, other_truth, other_initial, _, _) in runs.items():
        if (any(settings[key] != other[key] for key in pair_fields) or survey_settings(other, other_truth) != survey
                or truth.dtype != other_truth.dtype or initial.dtype != other_initial.dtype
                or not np.array_equal(truth, other_truth) or not np.array_equal(initial, other_initial)):
            raise ValueError(f"Unpaired observations, inputs, precision, versions or survey for {method}")

    rows = []
    truth64 = truth.astype(np.float64)
    initial_rmse = float(np.sqrt(np.mean((initial - truth64)**2)))
    with closing(FWI(dict(survey, n_shots=len(HELD_OUT_SHOTS)))) as heldout:
        heldout.sources[:, 0, 1] = torch.tensor(HELD_OUT_SHOTS)
        heldout.observed = heldout.forward(torch.from_numpy(truth), accuracy=8)
        fine_reference = refined_traces(heldout, truth, accuracy=8)
        fine_truth = refined_traces(heldout, truth, accuracy=4)
        for j, (stage, band) in enumerate(((1, 9), (3, 20), (4, 30))):
            reference_loss = heldout.loss(torch.from_numpy(truth), band)
            initial_loss = heldout.loss(torch.from_numpy(initial), band)
            for method in METHODS:
                _, progress, _, _, maps, macro_parameters = runs[method]
                record = progress[stage]
                model = maps[j]
                error = model - truth64
                rmse = float(np.sqrt(np.mean(error**2)))
                if not np.isclose(rmse, record["rmse"], rtol=1e-6, atol=1e-4):
                    raise ValueError(f"Saved stage RMSE does not match the {method} map at {band} Hz")
                prefix = progress[:stage+1]
                coarse_prediction = heldout.forward(torch.from_numpy(model))
                fine_prediction = refined_traces(heldout, model, accuracy=4)
                training_loss = record.get("post_update_loss", record.get("retained_loss"))
                if training_loss is None or not np.isfinite(training_loss):
                    raise ValueError(f"Missing or non-finite endpoint loss for {method} at {band} Hz")
                rows.append(dict(model=settings["model"], seed=settings["noise_seed"], method=method,
                                 band_hz=band, rmse_m_s=rmse, initial_rmse_m_s=initial_rmse,
                                 relative_l2=float(np.linalg.norm(error) / np.linalg.norm(truth64)),
                                 fitted_values=macro_parameters if stage == 1 else truth.size,
                                 training_loss=training_loss,
                                 heldout_clean_loss=band_loss(heldout, coarse_prediction, heldout.observed, band),
                                 heldout_30_hz=band_loss(heldout, coarse_prediction, heldout.observed, 30),
                                 fine_heldout_loss=band_loss(heldout, fine_prediction, fine_reference, band),
                                 fine_truth_loss=band_loss(heldout, fine_truth, fine_reference, band),
                                 initial_heldout_loss=initial_loss, truth_heldout_loss=reference_loss,
                                 fit_forward_shots=sum(p["forward_shots"] for p in prefix),
                                 fit_adjoint_shots=sum(p["adjoint_shots"] for p in prefix),
                                 reporting_forward_shots=sum(p["reporting_forward_shots"] for p in prefix),
                                 observation_sha256=settings["observation_sha256"]))
    return rows


def score(paths):
    groups = {}
    for path in paths:
        run = read_run(path)
        settings = run[0]
        group = groups.setdefault((settings["model"], settings["noise_seed"]), {})
        method = settings["method"]
        if method in group:
            raise ValueError(f"Duplicate {method} for {settings['model']}, seed {settings['noise_seed']}")
        group[method] = run
    return [row for key in sorted(groups) for row in compare_pair(groups[key])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="result archives or run directories")
    parser.add_argument("--output", required=True, type=Path, help="destination CSV")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = score(args.runs)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
