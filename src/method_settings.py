"""Gaussian row layout and fixed method choices. See README.md for the workflow."""

from dataclasses import dataclass

import torch

# Order in memory and in the saved NPZ files. The complexity charge counts this row.
GAUSSIAN_COLUMNS = ("z_m", "x_m", "log_width_along", "log_width_across", "angle_rad", "amplitude_m_s")
PARAMETERS_PER_GAUSSIAN = len(GAUSSIAN_COLUMNS)
DEFAULT_BAND_ITERATIONS = 325


@dataclass(frozen=True)
class GaussianSettings:
    # Work budget and placement. The width is the local wavelength divided by four.
    growth_interval: int = 10
    candidates_per_round: int = 6
    bridge_fraction: float = 0.01
    width_wavelength_divisor: float = 4.0

    # Relative to local speed at birth. The pruning screen has physical units.
    probe_speed_fraction: float = 0.05
    amplitude_cap_fraction: float = 0.30
    prune_amplitude_m_s: float = 5.0

    # Rates follow GAUSSIAN_COLUMNS. Centre rates are metres and angles are radians.
    adam_rates: tuple = (2.0, 2.0, 0.02, 0.02, 0.02, 10.0)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-20
    comparison_epsilons: int = 8


METHOD = GaussianSettings()


def minimum_width(local_speed, frequency, dx):
    """Lagged Gaussian width floor in metres, for one centre or a tensor of centres."""
    width = local_speed / (METHOD.width_wavelength_divisor * frequency)
    return width.clamp_min(dx) if isinstance(width, torch.Tensor) else max(dx, width)


def comparison_tolerance(before, after, dtype):
    """Allow for the different addition order in the gradient and forward loss sums."""
    return METHOD.comparison_epsilons * torch.finfo(dtype).eps * max(abs(before), abs(after))
