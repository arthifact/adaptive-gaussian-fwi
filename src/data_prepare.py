"""Load a velocity model from data/ and resample it for the run."""

from pathlib import Path

import numpy as np
import torch


def prepare_model(name, shape):
    """Load data/<name>.npy and area-resample it to `shape` (nz, nx)."""
    shape = tuple(int(n) for n in shape)
    root = Path(__file__).resolve().parents[1]
    velocity = torch.from_numpy(np.load(root / "data" / f"{name}.npy"))
    if not torch.isfinite(velocity).all() or (velocity <= 0).any():
        raise ValueError(f"{name}.npy must contain finite, positive velocities")
    # Interpolation expects [batch, channel, depth, x]; keep just the resized 2D field.
    velocity = torch.nn.functional.interpolate(
        velocity[None, None], size=shape, mode="area"
    )[0, 0]
    return velocity
