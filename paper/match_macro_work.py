"""Fixed lattice at 6 and 9 Hz with an adaptive run's per-band shot budget."""

import argparse
import hashlib
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from comparisons import fixed_gaussians as fixed
from paper.score_comparison import HELD_OUT_SHOTS, read_run, survey_settings
from src.simulation import FWI


def fixed_iterations(target, shots):
    """Updates that fit the budget: 1 endpoint forward + (forward + adjoint) each."""
    if target < shots or shots <= 0:
        raise ValueError("Budget must cover the fixed control's endpoint forward")
    return (target - shots) // (2 * shots)


def match_work(reference, output, rate=fixed.MACRO_RATE):
    """Rerun the fixed lattice with the adaptive shot budget and save to `output`."""
    reference = reference / "result.npz" if reference.is_dir() else reference
    settings, progress, truth, initial, _, _ = read_run(reference)
    if settings["method"] != "gaussian_background_pixels":
        raise ValueError("Expected a full adaptive reference run")
    fixed_truth, fixed_initial = fixed.load_models(settings["model"])
    if not np.array_equal(fixed_truth.numpy(), truth) or not np.array_equal(
        fixed_initial.numpy(), initial
    ):
        raise ValueError("The fixed control must use the same reference and ramp")
    survey = survey_settings(settings, truth)
    field = fixed.FixedField(fixed_initial)
    records, maps, parameters = [], [], {}
    with ThreadPoolExecutor(fixed.SHOTS) as pool:
        control = fixed.Survey(fixed_truth, settings["noise_seed"], pool)
        if fixed.digest(control.observed) != settings["observation_sha256"]:
            raise ValueError("The fixed control must use the same observations")
        for band, adaptive in zip(fixed.MACRO_BANDS, progress[:2], strict=True):
            target = adaptive["forward_shots"] + adaptive["adjoint_shots"]
            iterations = fixed_iterations(target, fixed.SHOTS)
            grid, record = fixed.fit_fixed_band(
                field, control, band, iterations, rate
            )
            total = record["forward_shots"] + record["adjoint_shots"]
            if not 0 <= target - total < 2 * fixed.SHOTS:
                raise ValueError("The recorded work does not match the intended budget")
            record.update(
                target_shot_count=target,
                iterations=iterations,
                fitted_values=field.amplitudes.numel(),
                rmse_m_s=float(
                    (grid.double() - fixed_truth.double()).square().mean().sqrt()
                ),
            )
            records.append(record)
            maps.append(grid.numpy().copy())
            parameters[f"amplitudes_{band}"] = field.amplitudes.numpy().copy()
            parameters[f"geometry_{band}"] = field.centers.numpy().copy()
    with closing(FWI(dict(survey, n_shots=len(HELD_OUT_SHOTS)))) as heldout:
        heldout.sources[:, 0, 1] = torch.tensor(HELD_OUT_SHOTS)
        heldout.observed = heldout.observe(fixed_truth)
        for grid, record in zip(maps, records, strict=True):
            record["heldout_clean_loss"] = heldout.loss(
                torch.from_numpy(grid), record["frequency"]
            )
    matched = dict(
        method="fixed_gaussians_matched_macro_work",
        model=settings["model"],
        noise_seed=settings["noise_seed"],
        noise_db=settings["noise_db"],
        macro_adam_rate=rate,
        survey=survey,
        input_sha256=settings["input_sha256"],
        observation_sha256=settings["observation_sha256"],
        adaptive_archive_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        result_selection={"gaussians": "best_measured_band_loss"},
        versions={name: version(name) for name in ("torch", "deepwave", "numpy")},
        source_sha256={
            str(path.relative_to(fixed.ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__).resolve(),
                Path(fixed.__file__),
                fixed.ROOT / "uv.lock",
                fixed.ROOT / "paper" / "score_comparison.py",
                fixed.ROOT / "src" / "simulation.py",
            )
        },
    )
    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".npz", delete=False
    ) as temp:
        staged = Path(temp.name)
    try:
        np.savez_compressed(
            staged,
            true=truth,
            initial=initial,
            macro=maps[-1],
            final=maps[-1],
            frequencies=fixed.MACRO_BANDS,
            stage_maps=np.stack(maps),
            settings=json.dumps(matched),
            progress=json.dumps(records),
            **parameters,
        )
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)
    fixed.save_preview(
        output.with_name("result.png"),
        truth,
        initial,
        fixed.MACRO_BANDS,
        maps,
        survey["bounds"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "adaptive_run", type=Path, help="adaptive run folder or its result.npz"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="default: comparisons/results/<case>/fixed_gaussians_matched/result.npz",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=fixed.MACRO_RATE,
        help="Adam step of the amplitudes",
    )
    args = parser.parse_args()
    run_folder = (
        args.adaptive_run if args.adaptive_run.is_dir() else args.adaptive_run.parent
    )
    if args.output is None:
        args.output = (
            fixed.ROOT
            / "comparisons"
            / "results"
            / run_folder.parent.name
            # Non-default rates go to their own folder, as in fixed_gaussians.py.
            / (
                "fixed_gaussians_matched"
                + ("" if args.rate == fixed.MACRO_RATE else f"_rate{args.rate:g}")
            )
            / "result.npz"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if (
        args.output.resolve()
        == (
            args.adaptive_run / "result.npz"
            if args.adaptive_run.is_dir()
            else args.adaptive_run
        ).resolve()
    ):
        parser.error("Use a separate output file for the matched control")
    torch.set_num_threads(1)
    match_work(args.adaptive_run, args.output, args.rate)
