"""Fixed background plus anisotropic Gaussian velocity corrections."""

import math

import torch

from .method_settings import PARAMETERS_PER_GAUSSIAN


def gaussian(dz, dx, long, short, angle):
    """Unit-height Gaussian; widths and offsets in metres, angle from x toward depth."""
    angle = torch.as_tensor(angle, dtype=dz.dtype, device=dz.device)
    along = (torch.cos(angle) * dx + torch.sin(angle) * dz) / long
    across = (torch.cos(angle) * dz - torch.sin(angle) * dx) / short
    return torch.exp(-0.5 * (along**2 + across**2))


class GaussianField:
    """Rows are [z, x, log width along, log width across, angle, amplitude].

    When a Gaussian is created the first axis is at least as wide as the second. The two
    log widths are optimized independently, so their ordering can change.
    Grid shape, dtype and device come from the background.
    """

    def __init__(self, background, dx):
        self.background = background.detach().clone()
        self.dx = float(dx)
        self.params = background.new_zeros((0, PARAMETERS_PER_GAUSSIAN))
        nz, nx = background.shape
        z = torch.arange(nz, dtype=background.dtype, device=background.device) * dx
        x = torch.arange(nx, dtype=background.dtype, device=background.device) * dx
        self.z, self.x = torch.meshgrid(z, x, indexing="ij")

    def __len__(self):
        return len(self.params)

    def validate_parameters(self, parameters, context):
        """Check a loaded or proposed table before its velocity is used for fitting."""
        if parameters.ndim != 2 or parameters.shape[1] != PARAMETERS_PER_GAUSSIAN:
            raise ValueError(f"Expected {PARAMETERS_PER_GAUSSIAN} columns in {context}")
        if not bool(torch.isfinite(parameters).all()):
            raise FloatingPointError(f"Non-finite Gaussian parameters in {context}")
        widths = parameters[:, 2:4].exp()
        if not bool((torch.isfinite(widths) & (widths > 0)).all()):
            raise FloatingPointError(f"Invalid Gaussian widths in {context}")

    def velocity(self, params=None):
        """Background plus all Gaussians in `params` (default: the stored table)."""
        parameters = self.params if params is None else params
        # Each parameter gets two grid axes so all Gaussians can be evaluated together.
        z, x, log_long, log_short, angle, amplitude = parameters.T[..., None, None]
        shapes = gaussian(
            self.z - z, self.x - x, log_long.exp(), log_short.exp(), angle
        )
        return self.background + (amplitude * shapes).sum(0)

    def params_with_gaussian(self, z, x, long, short, angle, amp):
        """Return the parameter table a proposed addition would store."""
        row = self.params.new_tensor(
            [[z, x, math.log(long), math.log(short), angle, amp]]
        )
        return torch.cat((self.params, row))

    def add(self, z, x, long, short, angle, amp):
        """Store one more Gaussian; `long` and `short` are widths in metres."""
        self.params = self.params_with_gaussian(z, x, long, short, angle, amp)

    def keep(self, mask):
        """Keep only the Gaussians where `mask` is true."""
        self.params = self.params[mask]
