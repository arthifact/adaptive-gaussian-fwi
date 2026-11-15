"""Parameter layout and fixed settings for the Gaussian stage."""

from dataclasses import dataclass

import torch

# Column order in memory and in saved NPZ files; the threshold charges per row.
GAUSSIAN_COLUMNS = (
    "z_m",
    "x_m",
    "log_width_along",
    "log_width_across",
    "angle_rad",
    "amplitude_m_s",
)
PARAMETERS_PER_GAUSSIAN = len(GAUSSIAN_COLUMNS)
DEFAULT_BAND_ITERATIONS = 325


@dataclass(frozen=True)
class GaussianSettings:
    """Fixed choices for the Gaussian stage: growth, amplitudes, pruning and Adam."""

    # Growth and placement.
    growth_interval: int = 10  # propose new Gaussians every 10th iteration
    candidates_per_round: int = 6  # try at most 6 new Gaussians per round
    bridge_fraction: float = 0.01  # cut gradient cells below 1% of their patch's peak
    width_wavelength_divisor: float = (
        4.0  # narrowest Gaussian: a quarter of the local wavelength
    )

    # Amplitudes, as fractions of the local speed where a Gaussian is added.
    probe_speed_fraction: float = 0.05  # first trial amplitude: 5% of local speed
    amplitude_cap_fraction: float = 0.30  # never start larger than 30% of local speed
    prune_amplitude_m_s: float = (
        5.0  # Gaussians weaker than 5 m/s are pruning candidates
    )

    # Adam step sizes, one per column of GAUSSIAN_COLUMNS: centres (m), log widths,
    # angle (rad) and amplitude (m/s).
    adam_rates: tuple = (2.0, 2.0, 0.02, 0.02, 0.02, 10.0)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-20
    comparison_epsilons: int = 8  # roundoff allowance when comparing two losses


METHOD = GaussianSettings()


def minimum_width(local_speed, frequency, dx):
    """Lagged Gaussian width floor in metres, for one centre or a tensor of centres."""
    width = local_speed / (METHOD.width_wavelength_divisor * frequency)
    return width.clamp_min(dx) if isinstance(width, torch.Tensor) else max(dx, width)


def comparison_tolerance(before, after, dtype):
    """Allow for the different addition order in the gradient and forward loss sums."""
    return (
        METHOD.comparison_epsilons
        * torch.finfo(dtype).eps
        * max(abs(before), abs(after))
    )
