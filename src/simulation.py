"""Survey, Deepwave forward model and the band-limited misfit.

forward simulates all shots, lowpass tapers the data below a cutoff, loss and
gradient give the normalized misfit and its derivative per cell (one adjoint
run per shot), and growth_threshold gives the BIC price per Gaussian.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import deepwave
import torch

from .method_settings import PARAMETERS_PER_GAUSSIAN

# Wave solver and signal-processing conventions for these experiments.
PREDICTION_ACCURACY = 4
PML_WIDTH_CELLS = 20
RICKER_PEAK_PERIODS = 1.5
MAX_SHOT_WORKERS = 8
TAPER_TRANSITION_FRACTION = 0.30
MINIMUM_NOISE_WINDOW_S = 0.30


class FWI:
    """Survey simulation and misfit; assign observed recordings before fitting."""

    def __init__(self, survey):
        self.shape = tuple(int(n) for n in survey["shape"])
        self.dx, self.dt = float(survey["dx"]), float(survey["dt"])
        self.nt, self.n_shots = int(survey["nt"]), int(survey["n_shots"])
        self.source_frequency = float(survey["source_frequency"])
        self.bounds = tuple(survey["bounds"])
        self.surface_velocity = float(survey["surface_velocity"])
        nx = self.shape[1]
        n_receivers = (
            nx if survey["n_receivers"] is None else int(survey["n_receivers"])
        )
        self.sources = torch.zeros(self.n_shots, 1, 2, dtype=torch.long)
        self.sources[:, 0, 1] = torch.linspace(0, nx - 1, self.n_shots).round().long()
        self.receivers = torch.zeros(1, n_receivers, 2, dtype=torch.long)
        self.receivers[0, :, 1] = torch.linspace(0, nx - 1, n_receivers).round().long()
        self.wavelet = deepwave.wavelets.ricker(
            self.source_frequency,
            self.nt,
            self.dt,
            RICKER_PEAK_PERIODS / self.source_frequency,
        ).reshape(1, 1, self.nt)
        self._pool = ThreadPoolExecutor(min(self.n_shots, MAX_SHOT_WORKERS))
        self._count_lock = Lock()
        self.forward_shots = self.adjoint_shots = 0
        self.observed = None
        # Growth threshold multiplier; 1 except in the sensitivity runs.
        self.threshold_scale = 1.0

    def close(self):
        """Finish pending shot work and release this survey's worker threads."""
        self._pool.shutdown()

    @property
    def observed(self):
        """Observed recordings (shots, receivers, time samples)."""
        return self._observed

    @observed.setter
    def observed(self, recordings):
        """New recordings; clears any noise energies given for the old ones."""
        self._observed = recordings
        self.noise_energy = {}

    def simulate(self, velocity, shot, receivers=None, accuracy=PREDICTION_ACCURACY):
        """Recordings of one shot at `receivers` (default: the surface line)."""
        with self._count_lock:
            self.forward_shots += 1
        if receivers is None:
            receivers = self.receivers
        result = deepwave.scalar(
            velocity,
            self.dx,
            self.dt,
            source_amplitudes=self.wavelet.to(velocity),
            source_locations=self.sources[shot : shot + 1],
            receiver_locations=receivers,
            accuracy=accuracy,
            pml_width=PML_WIDTH_CELLS,
            pml_freq=self.source_frequency,
            max_vel=self.bounds[1],
        )
        # Deepwave returns receiver recordings last; remove the single-shot axis.
        return result[-1][0]

    def forward(self, velocity, accuracy=PREDICTION_ACCURACY):
        """Generate clean recordings [shot, receiver, time]."""
        # Detach before starting worker threads; no_grad itself is thread-local.
        velocity = velocity.detach()
        recordings = self._pool.map(
            lambda s: self.simulate(velocity, s, accuracy=accuracy),
            range(self.n_shots),
        )
        return torch.stack(list(recordings))

    def add_noise(self, shots, snr_db, seed=0):
        """Add reproducible Gaussian noise with the source wavelet's spectrum."""
        generator = torch.Generator(device=shots.device).manual_seed(seed)
        white = torch.randn(
            shots.shape, dtype=shots.dtype, device=shots.device, generator=generator
        )
        n_fft = 2 * shots.shape[-1]
        white_spectrum = torch.fft.rfft(white, n_fft)
        wavelet_spectrum = torch.fft.rfft(self.wavelet[0, 0].to(shots), n_fft).abs()
        noise = torch.fft.irfft(white_spectrum * wavelet_spectrum, n_fft)
        noise = noise[..., : shots.shape[-1]]
        scaled_noise = noise * shots.norm() / noise.norm()
        return shots + scaled_noise / 10 ** (snr_db / 20)

    def validate_cutoff(self, frequency):
        """Raise ValueError unless 0 < frequency <= Nyquist."""
        nyquist = 0.5 / self.dt
        if not math.isfinite(frequency) or not 0 < float(frequency) <= nyquist:
            raise ValueError(f"Cutoff {frequency} Hz is not in (0, {nyquist}] Hz")

    def _lowpass_taper(self, recordings, frequency):
        """Taper: flat to 70% of the cutoff, cos^2 roll-off to zero at the cutoff."""
        self.validate_cutoff(frequency)
        frequency = float(frequency)
        dtype = (
            recordings.dtype
            if recordings.is_floating_point()
            else torch.get_default_dtype()
        )
        frequencies = torch.fft.rfftfreq(
            2 * recordings.shape[-1], self.dt, dtype=dtype, device=recordings.device
        )
        # Fade smoothly from 70% of the cutoff to zero at the cutoff.
        transition = TAPER_TRANSITION_FRACTION * frequency
        passband_end = (1 - TAPER_TRANSITION_FRACTION) * frequency
        rolloff = ((frequencies - passband_end) / transition).clamp(0, 1)
        taper = torch.cos(0.5 * torch.pi * rolloff) ** 2
        taper[frequencies >= frequency] = 0.0  # cos(pi/2) leaves a roundoff residue.
        taper[0] = 1.0  # DC is always passed, even for a sub-bin cutoff.
        return taper

    def _band_power_weights(self, frequency):
        """Power in the full spectrum represented by each real-FFT bin."""
        weights = self._lowpass_taper(self.observed, frequency).square()
        weights[1:-1] *= (
            2  # Include the negative-frequency partners, except at DC and Nyquist.
        )
        return weights

    def lowpass(self, recordings, frequency):
        """Low-pass recordings in time with the band taper (zero-padded FFT)."""
        n_fft = 2 * recordings.shape[-1]
        taper = self._lowpass_taper(recordings, frequency)
        spectrum = torch.fft.rfft(recordings, n_fft)
        filtered = torch.fft.irfft(spectrum * taper, n_fft)
        return filtered[..., : recordings.shape[-1]]

    def _data_energy(self, frequency):
        """Energy of the low-passed observed data, used to normalize the loss."""
        energy = (self.lowpass(self.observed, frequency) ** 2).sum()
        if not torch.isfinite(energy) or energy <= torch.finfo(energy.dtype).tiny:
            raise ValueError(
                "Observed data have no usable energy in this frequency band"
            )
        return energy

    def loss(self, velocity, frequency):
        """Normalized low-passed waveform misfit for `velocity` in this band."""
        data_energy = self._data_energy(frequency)
        residual = self.lowpass(self.forward(velocity) - self.observed, frequency)
        return float((residual**2).sum() / data_energy)

    def gradient(self, velocity, frequency):
        """Normalized misfit and its velocity gradient, summed over shots."""
        data_energy = self._data_energy(frequency)

        def shot_gradient(shot):
            """Misfit and velocity gradient for one shot."""
            shot_velocity = velocity.detach().requires_grad_()
            predicted = self.simulate(shot_velocity, shot)
            residual = self.lowpass(predicted - self.observed[shot], frequency)
            loss = (residual**2).sum()
            loss.backward()
            return loss.detach(), shot_velocity.grad

        losses, gradients = zip(*self._pool.map(shot_gradient, range(self.n_shots)))
        with self._count_lock:
            self.adjoint_shots += len(gradients)
        return float(sum(losses) / data_energy), sum(gradients) / data_energy

    def illumination(self, velocity, frequency):
        """Diagonal pseudo-Hessian used to rank candidates.

        Sum over shots of the squared second time derivative of the low-passed source
        wavefield in each cell (Shin et al., 2001).
        """
        self.validate_cutoff(frequency)
        velocity = velocity.detach()
        depth_cells = torch.arange(self.shape[0])
        horizontal_cells = torch.arange(self.shape[1])
        cells = torch.cartesian_prod(depth_cells, horizontal_cells)[None]

        def shot_energy(shot):
            """Squared second time derivative of the low-passed wavefield, one shot."""
            wavefield = self.simulate(velocity, shot, cells)
            filtered = self.lowpass(wavefield, frequency)
            second_difference = torch.diff(filtered, n=2, dim=-1)
            return (second_difference**2).sum(-1)

        with torch.no_grad():
            energies = self._pool.map(shot_energy, range(self.n_shots))
            energy = sum(energies).reshape(self.shape) / self.dt**4
            # The squared v-to-slowness-squared derivative contributes 4/v**6.
            # The common factor 4 cancels in the normalized illumination score.
            return energy / velocity.clamp(*self.bounds) ** 6

    def _noise_variance(self, frequency):
        """Noise variance per sample in this band.

        Uses the noise energy from main.py if given, otherwise estimates it from the
        pre-arrival part of each trace.
        """
        # Keys may be strings after a JSON round trip.
        energies = {
            float(cutoff): float(value) for cutoff, value in self.noise_energy.items()
        }
        if float(frequency) in energies:
            return energies[float(frequency)] / self.observed.numel()

        n_fft = 2 * self.nt
        weights = self._band_power_weights(frequency)

        # No arrival can beat straight-line distance / maximum speed.
        offsets = (
            torch.linalg.vector_norm((self.receivers - self.sources).double(), dim=-1)
            * self.dx
        )
        max_speed = max(self.bounds[1], self.surface_velocity)
        quiet_samples = (
            (offsets / (max_speed * self.dt)).floor().clamp_max(self.nt).long()
        )
        power, window_energy = torch.zeros_like(weights), 0.0
        minimum_samples = max(2, math.ceil(MINIMUM_NOISE_WINDOW_S / self.dt))
        for shot, receiver in zip(*torch.where(quiet_samples >= minimum_samples)):
            sample_count = int(quiet_samples[shot, receiver])
            window = torch.hann_window(
                sample_count, dtype=weights.dtype, device=weights.device
            )
            noise_segment = self.observed[shot, receiver, :sample_count] * window
            power += torch.fft.rfft(noise_segment, n_fft).abs() ** 2
            window_energy += float((window**2).sum())
        if not window_energy:
            raise ValueError(
                f"No pre-arrival noise window of 0.3 s at {frequency:g} Hz; "
                "set noise_energy[frequency] (0.0 for clean data)."
            )
        return float((weights * power).sum() / window_energy / n_fft)

    def growth_threshold(self, frequency):
        """BIC price for one more Gaussian, on the scale of the normalized loss.

        6 parameters * ln(n_eff) * noise variance per effective sample, divided by
        the band's data energy.
        n_eff is the sample count scaled by the taper's mean power gain, since
        low-passing correlates neighbouring samples. Zero for noise-free data.
        """
        data_energy = self._data_energy(frequency)
        noise_variance = self._noise_variance(frequency)
        if noise_variance == 0:
            return 0.0
        # Scale the recorded sample count by the mean power gain on the padded FFT grid.
        sample_count = self.observed.numel()
        weights = self._band_power_weights(frequency)
        effective_samples = (
            sample_count * float(weights.sum(dtype=torch.float64)) / (2 * self.nt)
        )
        if effective_samples <= 1:
            raise ValueError(
                f"Too few effective samples ({effective_samples:g}) at {frequency:g} Hz"
            )
        penalty = PARAMETERS_PER_GAUSSIAN * math.log(effective_samples)
        return (
            self.threshold_scale
            * penalty
            * noise_variance
            * (sample_count / effective_samples)
            / float(data_energy)
        )
