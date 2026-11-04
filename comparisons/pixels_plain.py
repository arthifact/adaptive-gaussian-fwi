"""Fit velocities directly on the grid at 6, 9, 13, 20 and 30 Hz."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from threading import Lock
from time import perf_counter

import deepwave
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
SHAPE, DX, DT, NT = (70, 70), 10., .001, 1000
SHOTS, SOURCE_HZ, SNR_DB = 5, 15., 30.
MACRO_BANDS, GRID_BANDS = (6, 9), (13, 20, 30)
MACRO_ITERATIONS, GRID_ITERATIONS = 325, 350
ADAM_RATE, ADAM_EPS = 10., 1e-20


def load_models(name):
    values = np.load(ROOT / "data" / f"{name}.npy", allow_pickle=False)
    if values.ndim != 2 or not values.size or values.dtype not in (np.float32, np.float64):
        raise ValueError("Expected a 2-D float32 or float64 velocity array")
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Velocities must be finite and positive")
    truth = torch.nn.functional.interpolate(torch.from_numpy(values)[None, None],
                                            size=SHAPE, mode="area")[0, 0]
    depth = torch.arange(SHAPE[0], dtype=truth.dtype)[:, None] * DX
    initial = (1500. + 1500. * depth / depth.max()).expand_as(truth).to(truth).clone()
    return truth, initial


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


class Survey:
    """Five surface shots, source-shaped noise and the normalized band loss."""

    def __init__(self, truth, seed, pool):
        self.pool = pool
        self.bounds = 1000., max(5500., float(truth.max()))
        self.lock = Lock()
        self.forwards = self.adjoints = 0
        self.sources = torch.zeros(SHOTS, 1, 2, dtype=torch.long)
        self.sources[:, 0, 1] = torch.linspace(0, truth.shape[1] - 1, SHOTS).round().long()
        self.receivers = torch.zeros(1, truth.shape[1], 2, dtype=torch.long)
        self.receivers[0, :, 1] = torch.arange(truth.shape[1])
        self.wavelet = deepwave.wavelets.ricker(SOURCE_HZ, NT, DT, 1.5 / SOURCE_HZ).reshape(1, 1, NT)
        clean = self.forward(truth, accuracy=8)
        generator = torch.Generator().manual_seed(seed)
        white = torch.randn(clean.shape, dtype=truth.dtype, generator=generator)
        source = torch.fft.rfft(self.wavelet[0, 0].to(truth), 2 * NT).abs()
        noise = torch.fft.irfft(torch.fft.rfft(white, 2 * NT) * source, 2 * NT)[..., :NT]
        norm = noise.norm()
        if not torch.isfinite(norm) or norm == 0:
            raise ValueError("Cannot generate nonzero source-shaped noise")
        self.observed = clean + (noise * clean.norm() / norm) / 10**(SNR_DB / 20)

    def shot(self, velocity, index, receivers=None, accuracy=4):
        with self.lock:
            self.forwards += 1
        waves = deepwave.scalar(velocity, DX, DT, source_amplitudes=self.wavelet.to(velocity),
                                source_locations=self.sources[index:index + 1],
                                receiver_locations=self.receivers if receivers is None else receivers,
                                accuracy=accuracy, pml_width=20, pml_freq=SOURCE_HZ, max_vel=self.bounds[1])
        return waves[-1][0]

    def forward(self, velocity, accuracy=4):
        recordings = self.pool.map(lambda s: self.shot(velocity.detach(), s, accuracy=accuracy), range(SHOTS))
        return torch.stack(list(recordings))

    def start_band(self, frequency):
        if not 0 < frequency <= .5 / DT:
            raise ValueError("Band cutoff must be positive and at or below Nyquist")
        bins = torch.fft.rfftfreq(2 * NT, DT, dtype=self.observed.dtype)
        rolloff = ((bins - .7 * frequency) / (.3 * frequency)).clamp(0, 1)
        self.taper = torch.cos(.5 * torch.pi * rolloff).square()
        self.taper[bins >= frequency] = 0.
        self.taper[0] = 1.
        self.energy = self.filter(self.observed).square().sum()
        if not torch.isfinite(self.energy) or self.energy <= torch.finfo(self.energy.dtype).tiny:
            raise ValueError("No usable observed data energy in this band")

    def filter(self, traces):
        spectrum = torch.fft.rfft(traces, 2 * NT) * self.taper
        return torch.fft.irfft(spectrum, 2 * NT)[..., :NT]

    def loss(self, velocity):
        residual = self.filter(self.forward(velocity) - self.observed)
        value = float(residual.square().sum() / self.energy)
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite waveform loss")
        return value

    def gradient(self, velocity):
        def one_shot(index):
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


METHOD_NAME = "pixels_plain"


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
    record = dict(stage="pixels", frequency=frequency, loss=losses, post_update_loss=endpoint,
                  seconds=seconds, forward_shots=forwards, adjoint_shots=adjoints,
                  reporting_forward_shots=survey.forwards-before[0]-forwards)
    return grid, record


def save_preview(path, truth, initial, frequencies, maps, bounds):
    panels = [("Simulation truth", truth), ("Starting ramp", initial)]
    panels.extend((f"Plain pixels\n{band:g} Hz", grid)
                  for band, grid in zip(frequencies, maps, strict=True))
    nz, nx = truth.shape
    extent = (-DX / 2, (nx - .5) * DX, (nz - .5) * DX, -DX / 2)
    fig = Figure(figsize=(3.1 * len(panels), 3.7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(panels))
    for ax, (title, grid) in zip(axes, panels, strict=True):
        image = ax.imshow(grid, interpolation="nearest", extent=extent, cmap="viridis",
                          vmin=bounds[0], vmax=bounds[1])
        ax.set(title=title, xlabel="Distance (m)")
    axes[0].set_ylabel("Depth (m)")
    fig.colorbar(image, ax=list(axes), label="Velocity (m/s)", shrink=.85)
    fig.savefig(path, dpi=200)


def run(model, seed, macro_only=False):
    torch.set_num_threads(1)
    truth, initial = load_models(model)
    grid_bands = () if macro_only else GRID_BANDS
    frequencies = (*MACRO_BANDS, *grid_bands)
    maps, progress = [], []
    with ThreadPoolExecutor(SHOTS) as pool:
        survey = Survey(truth, seed, pool)
        details = torch.nn.Parameter(torch.zeros_like(initial))
        for frequency in frequencies:
            iterations = MACRO_ITERATIONS if frequency in MACRO_BANDS else GRID_ITERATIONS
            grid, record = fit_grid_band(initial, details, survey, frequency, iterations)
            maps.append(grid.numpy().copy())
            progress.append(record)
        macro = maps[len(MACRO_BANDS)-1]
        for grid, stage in zip(maps, progress, strict=True):
            stage["rmse"] = float((torch.from_numpy(grid)-truth).square().mean().sqrt())
        settings = dict(method=METHOD_NAME, model=model, noise_seed=seed, noise_db=SNR_DB,
                        observation_sha256=digest(survey.observed), observation_shots=SHOTS,
                        result_selection={"pixels": "last_projected_adam_step"},
                        seconds_scope="band_setup_and_fitting_excluding_reporting",
                        model_transform="area_resample_whole_input_to_survey_grid",
                        input_sha256=hashlib.sha256((ROOT / "data" / f"{model}.npy").read_bytes()).hexdigest(),
                        versions={name: version(name) for name in ("torch", "deepwave", "numpy")},
                        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in (Path(__file__).resolve(), ROOT / "uv.lock")},
                        macro_frequencies=MACRO_BANDS, macro_iterations=MACRO_ITERATIONS,
                        pixel_frequencies=grid_bands, pixel_iterations=GRID_ITERATIONS,
                        survey=dict(shape=list(SHAPE), dx=DX, dt=DT, nt=NT, n_shots=SHOTS, n_receivers=SHAPE[1],
                                    source_frequency=SOURCE_HZ, bounds=survey.bounds))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    directory = ROOT / "outputs" / f"{METHOD_NAME}_{model}_seed{seed}_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    temp = directory / "result.tmp"
    with temp.open("wb") as stream:
        np.savez_compressed(stream, true=truth.numpy(), initial=initial.numpy(), macro=macro, final=maps[-1],
                            frequencies=frequencies, stage_maps=np.stack(maps), settings=json.dumps(settings),
                            progress=json.dumps(progress))
    temp.replace(directory / "result.npz")
    save_preview(directory / "result.png", truth.numpy(), initial.numpy(), frequencies, maps, survey.bounds)
    print(f"Saved {directory}", flush=True)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(p.stem for p in (ROOT / "data").glob("*.npy")), default="marmousi")
    parser.add_argument("--noise-seed", type=int, default=2)
    parser.add_argument("--macro-only", action="store_true", help="stop after 9 Hz")
    args = parser.parse_args()
    run(args.model, args.noise_seed, args.macro_only)
