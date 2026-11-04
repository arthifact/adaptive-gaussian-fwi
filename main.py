"""Fit a Gaussian macro model, freeze it, then refine the velocity on the grid."""

import argparse
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scripts.make_figures import plot_runs
from src.data_prepare import prepare_model
from src.gaussians import GaussianField
from src.gaussians_fit import fit_gaussians
from src.method_settings import DEFAULT_BAND_ITERATIONS, GAUSSIAN_COLUMNS, METHOD
from src.pixels import PixelField, fit_pixels
from src.simulation import FWI, MINIMUM_NOISE_WINDOW_S, PML_WIDTH_CELLS, PREDICTION_ACCURACY, TAPER_TRANSITION_FRACTION

# Survey and experiment choices. Gaussian method choices live in src/method_settings.py.
model = "marmousi"
survey = {
    "shape": (70, 70),         # depth cells, horizontal cells
    "dx": 10.0,               # metres per cell
    "n_shots": 5,
    "n_receivers": None,      # one receiver per surface cell
    "dt": 0.001,              # seconds per sample
    "nt": 1000,
    "source_frequency": 15.0,  # Ricker peak frequency, Hz
}
noise_db = 30                  # None uses clean observations.
noise_seed = 2
gaussian_frequencies = (6, 9)  # low-pass cutoffs, Hz
gaussian_iterations = DEFAULT_BAND_ITERATIONS
pixel_frequencies = (13, 20, 30)
pixel_iterations = 350
prune_gaussians = True
RAMP_M_S = (1500.0, 3000.0)
MIN_SPEED_M_S = 1000.0
MIN_UPPER_BOUND_M_S = 5500.0
OBSERVATION_ACCURACY = 8


def prepare_observations(problem, truth, seed):
    """Generate the synthetic data and record the actual added noise energy in each band."""
    for frequency in gaussian_frequencies:
        problem.validate_cutoff(frequency)
    clean = problem.forward(truth, accuracy=OBSERVATION_ACCURACY)
    observed = clean if noise_db is None else problem.add_noise(clean, noise_db, seed)
    problem.observed = observed
    noise = observed - clean
    problem.noise_energy = {
        f: float(problem.lowpass(noise, f).square().sum()) for f in dict.fromkeys(gaussian_frequencies)
    }


def fit_stages(field, problem, truth, *, prune):
    """Carry the retained parameters from each band into the next band."""
    backgrounds, progress, parameters = [], [], {}
    for stage_index, frequency in enumerate(gaussian_frequencies):
        started = perf_counter()
        before = problem.forward_shots, problem.adjoint_shots
        decisions = []
        losses = fit_gaussians(field, problem, frequency, iterations=gaussian_iterations,
                               prune=prune, decisions=decisions)
        snapshot = field.velocity().clamp(*problem.bounds).detach().clone()
        seconds = perf_counter() - started
        backgrounds.append((frequency, snapshot))
        parameters[f"gaussian_params_{stage_index}"] = field.params.detach().numpy().copy()
        error = snapshot - truth
        progress.append({
            "stage": "gaussians", "frequency": frequency, "loss": losses, "decisions": decisions,
            "forward_shots": problem.forward_shots - before[0], "adjoint_shots": problem.adjoint_shots - before[1],
            "reporting_forward_shots": problem.n_shots if losses else 0,
            "post_update_loss": problem.loss(snapshot, frequency) if losses else None,
            "seconds": seconds, "gaussians": len(field),
            "birth_threshold": problem.birth_threshold(frequency) if losses else None,
            "relative_l2": float(error.norm() / truth.norm()),
            "rmse": float(error.square().mean().sqrt()),
        })
    return backgrounds, progress, parameters


def refine_grid(background, problem, truth, bands):
    """Hold the retained macro map fixed and fit one additive correction per cell."""
    pixels = PixelField(background)
    models, progress = [], []
    for frequency in bands:
        started = perf_counter()
        before = problem.forward_shots, problem.adjoint_shots
        losses = fit_pixels(pixels, problem, frequency, iterations=pixel_iterations)
        snapshot = pixels.velocity().detach().clone()
        seconds = perf_counter() - started
        models.append(snapshot)
        error = snapshot - truth
        forwards, adjoints = problem.forward_shots - before[0], problem.adjoint_shots - before[1]
        progress.append(dict(stage="pixels", frequency=frequency, loss=losses,
                             seconds=seconds,
                             post_update_loss=problem.loss(snapshot, frequency) if losses else None,
                             forward_shots=forwards, adjoint_shots=adjoints,
                             reporting_forward_shots=problem.n_shots if losses else 0,
                             relative_l2=float(error.norm()/truth.norm()), rmse=float(error.square().mean().sqrt())))
    return models, progress


def run_experiment(model_name, seed, *, macro_only=False, prune=None):
    """Run the published experiment and return the directory containing the results."""
    torch.set_num_threads(1)  # Each source shot already runs in its own worker.
    if prune is None:
        prune = prune_gaussians
    truth = prepare_model(model_name, survey["shape"])
    # These bounds use the synthetic reference and must be disclosed with the results.
    settings_survey = dict(survey, bounds=(MIN_SPEED_M_S, max(MIN_UPPER_BOUND_M_S, float(truth.max()))),
                           surface_velocity=float(truth[0].max()))
    depth = torch.arange(truth.shape[0], dtype=truth.dtype, device=truth.device)[:, None] * settings_survey["dx"]
    top, bottom = RAMP_M_S
    initial = (top + (bottom - top) * depth / depth.max()).expand_as(truth).clone()
    field = GaussianField(initial, settings_survey["dx"])
    grid_bands = () if macro_only else pixel_frequencies
    with closing(FWI(settings_survey)) as problem:
        for frequency in grid_bands:
            problem.validate_cutoff(frequency)
        prepare_observations(problem, truth, seed)
        observation_hash = hashlib.sha256(problem.observed.detach().contiguous().numpy().tobytes()).hexdigest()
        backgrounds, progress, parameters = fit_stages(field, problem, truth, prune=prune)
        macro = field.velocity().clamp(*problem.bounds).detach()
        pixels, grid_progress = refine_grid(macro, problem, truth, grid_bands)
        progress.extend(grid_progress)
        noise_energy = problem.noise_energy
    final = pixels[-1] if pixels else macro

    root = Path(__file__).resolve().parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = root / "outputs" / f"adaptive_{model_name}_seed{seed}_{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "method": "gaussian_background_pixels" if grid_bands else "adaptive_gaussian_macro",
        "placement": "signed_region_moments", "observation_sha256": observation_hash,
        "gaussian_columns": GAUSSIAN_COLUMNS, "gaussian_method_settings": asdict(METHOD),
        "region_relative_floor": METHOD.bridge_fraction,
        "birth_amplitude_rule": "lowest_measured_loss_probe_wins_ties",
        "model": model_name, "survey": settings_survey, "noise_db": noise_db, "noise_seed": seed,
        "noise_energy": noise_energy, "noise_energy_source": "synthetic_added_noise",
        "observation_accuracy": OBSERVATION_ACCURACY, "prediction_accuracy": PREDICTION_ACCURACY,
        "pml_width_cells": PML_WIDTH_CELLS, "lowpass_transition_fraction": TAPER_TRANSITION_FRACTION,
        "minimum_noise_window_s": MINIMUM_NOISE_WINDOW_S,
        "birth_threshold_method": "sampled_filter_power", "gaussian_pruning": prune,
        "pruning_rule": "loss_checked_weak_batch",
        "gaussian_frequencies": gaussian_frequencies, "gaussian_iterations": gaussian_iterations,
        "pixel_frequencies": grid_bands, "pixel_iterations": pixel_iterations,
        "initial_velocity": RAMP_M_S,
        "result_selection": {"gaussians": "best_measured_loss_plus_complexity_per_band",
                             "pixels": "last_projected_adam_step"},
        "seconds_scope": "band_setup_and_fitting_excluding_reporting",
        "model_transform": "area_resample_whole_input_to_survey_grid",
        "input_sha256": hashlib.sha256((root / "data" / f"{model_name}.npy").read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ("torch", "deepwave", "numpy")},
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "main.py", root / "scripts" / "make_figures.py", root / "uv.lock", *sorted((root / "src").glob("*.py")))
        },
    }
    saved = torch.stack([grid for _, grid in backgrounds]) if backgrounds else truth.new_empty((0, *truth.shape))
    saved_pixels = torch.stack(pixels) if pixels else truth.new_empty((0, *truth.shape))
    temp = output / "result.tmp"
    with temp.open("wb") as stream:
        np.savez_compressed(stream, true=truth.numpy(), initial=initial.numpy(), macro=macro.numpy(),
                            final=final.numpy(), gaussian_frequencies=gaussian_frequencies,
                            gaussian_backgrounds=saved.numpy(), pixel_frequencies=grid_bands,
                            pixel_models=saved_pixels.numpy(), settings=json.dumps(settings),
                            progress=json.dumps(progress), **parameters)
    temp.replace(output / "result.npz")
    plot_runs([output])
    print(f"Saved {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    data_dir = Path(__file__).resolve().parent / "data"
    parser.add_argument("--model", choices=sorted(path.stem for path in data_dir.glob("*.npy")), default=model,
                        help="velocity model from data/")
    parser.add_argument("--noise-seed", type=int, default=noise_seed)
    parser.add_argument("--macro-only", action="store_true", help="stop after the Gaussian bands")
    parser.add_argument("--pruning", action=argparse.BooleanOptionalAction, default=prune_gaussians,
                        help="enable or disable loss-checked weak-atom pruning")
    args = parser.parse_args()
    run_experiment(args.model, args.noise_seed, macro_only=args.macro_only, prune=args.pruning)
