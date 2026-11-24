"""Fixed Gaussian lattice at 6 and 9 Hz, then pixel refinement at 13, 20 and 30 Hz.

MACRO_RATE, the Adam step of the lattice amplitudes, was chosen on Marmousi seed 1
only (paper/run_experiments.sh tune).

Standalone on purpose: it has its own survey, simulation and loss and imports
nothing from src/. tests/test_comparisons.py checks that the data, loss and
gradient match src/simulation.py.
"""

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from threading import Lock
from time import perf_counter

import deepwave
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
SHAPE, DX, DT, NT = (70, 70), 10.0, 0.001, 1000
SHOTS, SOURCE_HZ, SNR_DB = 5, 15.0, 20.0
MACRO_BANDS, GRID_BANDS = (6, 9), (13, 20, 30)
MACRO_ITERATIONS, GRID_ITERATIONS = 325, 350
ADAM_RATE, ADAM_EPS = 10.0, 1e-20
MACRO_RATE = 1.0  # m/s per Adam step of a lattice amplitude


def load_models(name):
    """Resampled true model from data/<name>.npy and the 1.5-3 km/s starting ramp."""
    values = np.load(ROOT / "data" / f"{name}.npy", allow_pickle=False)
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Velocities must be finite and positive")
    truth = torch.nn.functional.interpolate(
        torch.from_numpy(values)[None, None], size=SHAPE, mode="area"
    )[0, 0]
    depth = torch.arange(SHAPE[0], dtype=truth.dtype)[:, None] * DX
    initial = (1500.0 + 1500.0 * depth / depth.max()).expand_as(truth).to(truth).clone()
    return truth, initial


def digest(tensor):
    """SHA-256 of a tensor's bytes."""
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def tolerance(before, after, dtype):
    """Roundoff allowance for comparing two losses."""
    return 8 * torch.finfo(dtype).eps * max(abs(before), abs(after))


class Survey:
    """Five surface shots, white noise and the normalized band loss."""

    def __init__(self, truth, seed, pool):
        self.pool = pool
        self.bounds = 1000.0, 7000.0  # the same for every model
        self.lock = Lock()
        self.forwards = self.adjoints = 0
        self.sources = torch.zeros(SHOTS, 1, 2, dtype=torch.long)
        self.sources[:, 0, 1] = (
            torch.linspace(0, truth.shape[1] - 1, SHOTS).round().long()
        )
        self.receivers = torch.zeros(1, truth.shape[1], 2, dtype=torch.long)
        self.receivers[0, :, 1] = torch.arange(truth.shape[1])
        self.wavelet = deepwave.wavelets.ricker(
            SOURCE_HZ, NT, DT, 1.5 / SOURCE_HZ
        ).reshape(1, 1, NT)
        clean = self.observe(truth)
        generator = torch.Generator().manual_seed(seed)
        white = torch.randn(clean.shape, dtype=truth.dtype, generator=generator)
        self.observed = clean + white * clean.square().mean().sqrt() / 10 ** (
            SNR_DB / 20
        )

    def observe(self, velocity):
        """Clean recordings from a grid of half the spacing, 8th-order differences."""
        fine = torch.nn.functional.interpolate(
            velocity.detach()[None, None],
            size=tuple(2 * n - 1 for n in velocity.shape),
            mode="bilinear",
            align_corners=True,
        )[0, 0].contiguous()

        def one_shot(index):
            return deepwave.scalar(
                fine,
                DX / 2,
                DT,
                # Same point source: a quarter of the cell area needs 4x the amplitude.
                source_amplitudes=4 * self.wavelet.to(fine),
                source_locations=2 * self.sources[index : index + 1],
                receiver_locations=2 * self.receivers,
                accuracy=8,
                pml_width=40,
                pml_freq=SOURCE_HZ,
                max_vel=self.bounds[1],
            )[-1][0]

        return torch.stack(list(self.pool.map(one_shot, range(SHOTS))))

    def shot(self, velocity, index, receivers=None, accuracy=4):
        """Recordings of one shot; counts the forward solve."""
        with self.lock:
            self.forwards += 1
        waves = deepwave.scalar(
            velocity,
            DX,
            DT,
            source_amplitudes=self.wavelet.to(velocity),
            source_locations=self.sources[index : index + 1],
            receiver_locations=self.receivers if receivers is None else receivers,
            accuracy=accuracy,
            pml_width=20,
            pml_freq=SOURCE_HZ,
            max_vel=self.bounds[1],
        )
        return waves[-1][0]

    def forward(self, velocity, accuracy=4):
        """Recordings of every shot, computed in parallel."""
        recordings = self.pool.map(
            lambda s: self.shot(velocity.detach(), s, accuracy=accuracy), range(SHOTS)
        )
        return torch.stack(list(recordings))

    def start_band(self, frequency):
        """Set the low-pass taper and the observed data energy for a new band."""
        if not 0 < frequency <= 0.5 / DT:
            raise ValueError("Band cutoff must be positive and at or below Nyquist")
        bins = torch.fft.rfftfreq(2 * NT, DT, dtype=self.observed.dtype)
        rolloff = ((bins - 0.7 * frequency) / (0.3 * frequency)).clamp(0, 1)
        self.taper = torch.cos(0.5 * torch.pi * rolloff).square()
        self.taper[bins >= frequency] = 0.0
        self.taper[0] = 1.0
        self.energy = self.filter(self.observed).square().sum()
        if (
            not torch.isfinite(self.energy)
            or self.energy <= torch.finfo(self.energy.dtype).tiny
        ):
            raise ValueError("No usable observed data energy in this band")

    def filter(self, traces):
        """Low-pass filter traces along time with the current band taper."""
        spectrum = torch.fft.rfft(traces, 2 * NT) * self.taper
        return torch.fft.irfft(spectrum, 2 * NT)[..., :NT]

    def loss(self, velocity):
        """Normalized band misfit for `velocity`."""
        residual = self.filter(self.forward(velocity) - self.observed)
        value = float(residual.square().sum() / self.energy)
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite waveform loss")
        return value

    def gradient(self, velocity):
        """Normalized band misfit and its velocity gradient, summed over shots."""

        def one_shot(index):
            """Misfit and velocity gradient for one shot."""
            model = velocity.detach().requires_grad_()
            residual = self.filter(self.shot(model, index) - self.observed[index])
            loss = residual.square().sum()
            loss.backward()
            return loss.detach(), model.grad

        losses, gradients = zip(*self.pool.map(one_shot, range(SHOTS)))
        self.adjoints += len(gradients)
        loss, gradient = float(sum(losses) / self.energy), sum(gradients) / self.energy
        if not math.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite waveform loss or gradient")
        return loss, gradient


METHOD_NAME = "fixed_gaussians"


def fit_grid_band(background, details, survey, frequency, iterations):
    """Fit the grid at one frequency, carrying cell corrections across bands."""
    started = perf_counter()
    before = survey.forwards, survey.adjoints
    survey.start_band(frequency)
    optimizer = torch.optim.Adam([details], lr=ADAM_RATE, eps=ADAM_EPS)
    losses = []
    for _ in range(iterations):
        current = background + details.detach()
        loss, gradient = survey.gradient(current)
        losses.append(loss)
        optimizer.zero_grad(set_to_none=True)
        details.grad = gradient
        optimizer.step()
        with torch.no_grad():
            if not torch.isfinite(details).all():
                raise FloatingPointError("Non-finite grid update")
            details.clamp_(survey.bounds[0] - background, survey.bounds[1] - background)
    grid = (background + details.detach()).clone()
    seconds = perf_counter() - started
    forwards = survey.forwards - before[0]
    adjoints = survey.adjoints - before[1]
    endpoint = survey.loss(grid)
    record = dict(
        stage="pixels",
        frequency=frequency,
        loss=losses,
        post_update_loss=endpoint,
        seconds=seconds,
        forward_shots=forwards,
        adjoint_shots=adjoints,
        reporting_forward_shots=survey.forwards - before[0] - forwards,
    )
    return grid, record


def refine_grid(background, survey, bands, iterations=GRID_ITERATIONS):
    """Start from zero cell corrections on the frozen Gaussian field."""
    details = torch.nn.Parameter(torch.zeros_like(background))
    grids, records = [], []
    for frequency in bands:
        grid, record = fit_grid_band(background, details, survey, frequency, iterations)
        grids.append(grid.numpy().copy())
        records.append(record)
    return grids, records


class FixedField:
    """Immutable isotropic shapes, with one adjustable amplitude per center."""

    def __init__(self, initial):
        self.initial = initial
        self.banks = []
        self.centers = initial.new_empty(
            (0, 3)
        )  # (depth, horizontal position, width), metres
        self.amplitudes = initial.new_empty((0,))

    def extend(self, band):
        """Add isotropic Gaussians for `band`, quarter-wavelength spacing and width."""
        spacing = float(torch.quantile(self.initial.flatten(), 0.5)) / (4 * band)
        axes = []
        for cells in self.initial.shape:
            extent = (cells - 1) * DX
            count = max(1, math.ceil(extent / spacing))
            axes.append([(i + 0.5) * extent / count for i in range(count)])
        rows = []
        for z in axes[0]:
            for x in axes[1]:
                speed = float(self.initial[round(z / DX), round(x / DX)])
                rows.append((z, x, max(DX, speed / (4 * band))))
        geometry = self.initial.new_tensor(rows)
        z, x = torch.meshgrid(
            *(
                torch.arange(n, dtype=self.initial.dtype) * DX
                for n in self.initial.shape
            ),
            indexing="ij",
        )
        cz, cx, width = geometry.T[..., None, None]
        shapes = torch.exp(
            -0.5 * (((x - cx) / width).square() + ((z - cz) / width).square())
        )
        self.banks.append(shapes.flatten(1).contiguous())
        self.centers = torch.cat((self.centers, geometry))
        self.amplitudes = torch.cat(
            (self.amplitudes, geometry.new_zeros(len(geometry)))
        )

    def velocity(self, amplitudes=None):
        """Starting ramp plus the weighted lattice shapes."""
        amplitudes = self.amplitudes if amplitudes is None else amplitudes
        values = self.initial.flatten()
        offset = 0
        # Sum each bank separately so an all-zero bank leaves the field unchanged.
        for bank in self.banks:
            values = values + torch.mv(bank.T, amplitudes[offset : offset + len(bank)])
            offset += len(bank)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite fixed Gaussian velocity")
        return values.reshape_as(self.initial)


def fit_fixed_band(
    field, survey, frequency, iterations=MACRO_ITERATIONS, rate=MACRO_RATE
):
    """Keep all shapes fixed and keep the best measured amplitudes in this band."""
    started = perf_counter()
    before = survey.forwards, survey.adjoints
    field.extend(frequency)
    survey.start_band(frequency)
    amplitudes = torch.nn.Parameter(field.amplitudes.clone())
    optimizer = torch.optim.Adam([amplitudes], lr=rate, eps=ADAM_EPS)
    bank = torch.cat(field.banks)
    best, best_loss = amplitudes.detach().clone(), None
    losses = []
    for _ in range(iterations):
        raw = field.velocity(amplitudes.detach())
        loss, gradient = survey.gradient(raw.clamp(*survey.bounds))
        losses.append(loss)
        if best_loss is None or best_loss - loss > tolerance(
            best_loss, loss, raw.dtype
        ):
            best, best_loss = amplitudes.detach().clone(), loss
        active = gradient * ((raw >= survey.bounds[0]) & (raw <= survey.bounds[1]))
        optimizer.zero_grad(set_to_none=True)
        amplitudes.grad = bank @ active.flatten()
        optimizer.step()
        if not torch.isfinite(amplitudes).all():
            raise FloatingPointError("Non-finite fixed amplitudes")
    endpoint = survey.loss(field.velocity(amplitudes.detach()).clamp(*survey.bounds))
    if best_loss is None or best_loss - endpoint > tolerance(
        best_loss, endpoint, field.initial.dtype
    ):
        best, best_loss = amplitudes.detach().clone(), endpoint
    field.amplitudes = best
    grid = field.velocity().clamp(*survey.bounds)
    record = dict(
        stage="gaussians",
        frequency=frequency,
        loss=losses,
        retained_loss=best_loss,
        seconds=perf_counter() - started,
        forward_shots=survey.forwards - before[0],
        adjoint_shots=survey.adjoints - before[1],
        reporting_forward_shots=0,
    )
    return grid, record


def save_preview(path, truth, initial, frequencies, maps, bounds):
    """Save a PNG of the truth, the starting ramp and every stage map."""
    panels = [("Simulation truth", truth), ("Starting ramp", initial)]
    for band, grid in zip(frequencies, maps, strict=True):
        label = "Fixed Gaussians" if band in MACRO_BANDS else "Grid refinement"
        panels.append((f"{label}\n{band:g} Hz", grid))
    nz, nx = truth.shape
    extent = (-DX / 2, (nx - 0.5) * DX, (nz - 0.5) * DX, -DX / 2)
    fig = Figure(figsize=(3.1 * len(panels), 3.7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(panels))
    for ax, (title, grid) in zip(axes, panels, strict=True):
        image = ax.imshow(
            grid,
            interpolation="nearest",
            extent=extent,
            cmap="viridis",
            vmin=bounds[0],
            vmax=bounds[1],
        )
        ax.set(title=title, xlabel="Distance (m)")
    axes[0].set_ylabel("Depth (m)")
    fig.colorbar(image, ax=list(axes), label="Velocity (m/s)", shrink=0.85)
    fig.savefig(path, dpi=200)


def run(model, seed, macro_only=False, rate=MACRO_RATE):
    """Run one model and seed; save to comparisons/results/<model>-seed<seed>/."""
    torch.set_num_threads(1)
    truth, initial = load_models(model)
    grid_bands = () if macro_only else GRID_BANDS
    frequencies = (*MACRO_BANDS, *grid_bands)
    maps, progress, parameters = [], [], {}
    with ThreadPoolExecutor(SHOTS) as pool:
        survey = Survey(truth, seed, pool)
        field = FixedField(initial)
        for frequency in MACRO_BANDS:
            grid, record = fit_fixed_band(field, survey, frequency, rate=rate)
            maps.append(grid.numpy().copy())
            progress.append(record)
            parameters[f"amplitudes_{frequency}"] = field.amplitudes.numpy().copy()
            parameters[f"geometry_{frequency}"] = field.centers.numpy().copy()
        macro = maps[-1]
        grids, records = refine_grid(torch.from_numpy(macro), survey, grid_bands)
        maps.extend(grids)
        progress.extend(records)
        for grid, stage in zip(maps, progress, strict=True):
            stage["rmse"] = float(
                (torch.from_numpy(grid) - truth).square().mean().sqrt()
            )
        settings = dict(
            method=METHOD_NAME,
            model=model,
            noise_seed=seed,
            noise_db=SNR_DB,
            observation_sha256=digest(survey.observed),
            observation_shots=SHOTS,
            result_selection={
                "gaussians": "best_measured_band_loss",
                "pixels": "last_projected_adam_step",
            },
            seconds_scope="band_setup_and_fitting_excluding_reporting",
            model_transform="area_resample_whole_input_to_survey_grid",
            input_sha256=hashlib.sha256(
                (ROOT / "data" / f"{model}.npy").read_bytes()
            ).hexdigest(),
            versions={name: version(name) for name in ("torch", "deepwave", "numpy")},
            source_sha256={
                str(path.relative_to(ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in (Path(__file__).resolve(), ROOT / "uv.lock")
            },
            macro_frequencies=MACRO_BANDS,
            macro_iterations=MACRO_ITERATIONS,
            macro_adam_rate=rate,
            pixel_frequencies=grid_bands,
            pixel_iterations=GRID_ITERATIONS,
            survey=dict(
                shape=list(SHAPE),
                dx=DX,
                dt=DT,
                nt=NT,
                n_shots=SHOTS,
                n_receivers=SHAPE[1],
                source_frequency=SOURCE_HZ,
                bounds=survey.bounds,
            ),
        )
    folder = (
        ROOT
        / "comparisons"
        / "results"
        / f"{model}-seed{seed}"
        # Non-default rates go to their own folder, e.g. fixed_gaussians_rate3/.
        / (
            METHOD_NAME
            + ("" if rate == MACRO_RATE else f"_rate{rate:g}")
            + ("_macro" if macro_only else "")
        )
    )
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "result.npz"
    temp = output.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(
            stream,
            true=truth.numpy(),
            initial=initial.numpy(),
            macro=macro,
            final=maps[-1],
            frequencies=frequencies,
            stage_maps=np.stack(maps),
            settings=json.dumps(settings),
            progress=json.dumps(progress),
            **parameters,
        )
    temp.replace(output)
    save_preview(
        folder / "result.png",
        truth.numpy(),
        initial.numpy(),
        frequencies,
        maps,
        survey.bounds,
    )
    print(f"Saved {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(p.stem for p in (ROOT / "data").glob("*.npy")),
        default="marmousi",
    )
    parser.add_argument("--noise-seed", type=int, default=2)
    parser.add_argument("--macro-only", action="store_true", help="stop after 9 Hz")
    parser.add_argument(
        "--rate", type=float, default=MACRO_RATE, help="Adam step of the amplitudes"
    )
    args = parser.parse_args()
    if not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("--rate must be finite and positive")
    run(args.model, args.noise_seed, args.macro_only, args.rate)
