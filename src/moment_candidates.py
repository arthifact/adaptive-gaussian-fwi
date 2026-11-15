"""Candidate Gaussians from the gradient.

The gradient is split into connected same-sign patches, after cutting weak
links below 1% of each patch's peak (significant_components). Each patch gives
one Gaussian with the patch's weighted centre and covariance (moment_proposal).
Candidates are ranked by predicted loss decrease, and the best six are returned
(MomentCandidates.propose).
"""

import math
from time import perf_counter

import numpy as np
import torch

from .gaussians import gaussian
from .method_settings import METHOD, minimum_width

REGION_RELATIVE_FLOOR = METHOD.bridge_fraction


def signed_components(values):
    """4-connected patches of same-sign cells, as arrays of (z, x) indices.

    Zero cells belong to no patch. Patches are ordered by their first cell in a
    row-major scan, so the output is deterministic.
    """
    nz, nx = values.shape
    seen = np.zeros(values.shape, dtype=bool)
    groups = []
    for z0 in range(nz):
        for x0 in range(nx):
            if seen[z0, x0] or values[z0, x0] == 0:
                continue
            sign = values[z0, x0] > 0
            stack, points = [(z0, x0)], []
            seen[z0, x0] = True
            while stack:
                z, x = stack.pop()
                points.append((z, x))
                for zz, xx in ((z - 1, x), (z + 1, x), (z, x - 1), (z, x + 1)):
                    if (
                        0 <= zz < nz
                        and 0 <= xx < nx
                        and not seen[zz, xx]
                        and values[zz, xx] != 0
                        and (values[zz, xx] > 0) == sign
                    ):
                        seen[zz, xx] = True
                        stack.append((zz, xx))
            groups.append(np.asarray(points, dtype=np.int64))
    return groups


def significant_components(values):
    """Cut weak bridges at 1% of each original signed region's peak, then reconnect.

    The cutoff is local to an original component: a separate weak region is not
    suppressed by a strong region elsewhere. Full gradients still score shapes.
    """
    supported = values.copy()
    for indices in signed_components(values):
        z, x = indices.T
        magnitude = np.abs(values[z, x])
        weak = magnitude <= REGION_RELATIVE_FLOOR * magnitude.max()
        supported[z[weak], x[weak]] = 0
    return signed_components(supported)


def moment_proposal(indices, gradient, velocity, dx, frequency):
    """Gaussian matching one gradient patch.

    Centre from the |gradient|-weighted mean, widths and angle from the eigen-
    decomposition of the weighted covariance. Widths are floored at a quarter
    wavelength or one cell.
    """
    weights = np.abs(gradient[indices[:, 0], indices[:, 1]])
    mass = float(weights.sum())
    weights = weights / mass
    points = indices[:, ::-1] * dx  # covariance axes are (x, z)
    mean = (points * weights[:, None]).sum(axis=0)
    offsets = points - mean
    covariance = (offsets * weights[:, None]).T @ offsets
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Python round is also used by the waveform admission code.
    iz, ix = round(float(mean[1]) / dx), round(float(mean[0]) / dx)
    minimum = minimum_width(float(velocity[iz, ix]), frequency, dx)
    widths = np.maximum(np.sqrt(np.maximum(eigenvalues[::-1], 0)), minimum)
    angle = math.atan2(eigenvectors[1, -1], eigenvectors[0, -1]) % math.pi
    proposal = dict(
        z=float(mean[1]),
        x=float(mean[0]),
        long=float(widths[0]),
        short=float(widths[1]),
        angle=angle,
    )
    detail = dict(
        cells=len(indices),
        sign=int(np.sign(gradient[tuple(indices[0])])),
        gradient_mass=mass,
        covariance_xz_m2=covariance.tolist(),
        eigenvalues_m2=eigenvalues.tolist(),
        nearest_cell=[iz, ix],
        minimum_width_m=minimum,
    )
    return proposal, detail


class MomentCandidates:
    """Construct one shape per signed region and rank its predicted usefulness.

    Shapes use the field dtype; moments and score reductions use float64.
    The waveform fitter chooses amplitudes after geometry is proposed.
    """

    def __init__(self, field):
        self.z, self.x, self.dx = field.z, field.x, field.dx
        self.shape = tuple(field.background.shape)
        self.rounds, self.stats = [], {}

    @torch.no_grad()
    def propose(
        self,
        gradient,
        illumination,
        velocity,
        frequency,
        max_candidates=METHOD.candidates_per_round,
    ):
        """Up to `max_candidates` shapes from the gradient patches, best first."""
        start = perf_counter()

        # Out-of-place division preserves caller tensors even if already float64.
        g = gradient.detach().double()
        gmax = float(g.abs().max())
        if gmax:
            g = g / gmax
        h = illumination.detach().double()
        hmax = float(h.max())
        if hmax:
            h = h / hmax
        values = gradient.detach().double().cpu().numpy()
        current_velocity = velocity.detach().double().cpu().numpy()
        groups = significant_components(values)
        # Smallest resolvable curvature at this illumination scale.
        floor = math.ulp(float(h.sum()))
        records = []
        for region, indices in enumerate(groups):
            proposal, detail = moment_proposal(
                indices, values, current_velocity, self.dx, float(frequency)
            )
            shape = gaussian(
                self.z - proposal["z"],
                self.x - proposal["x"],
                proposal["long"],
                proposal["short"],
                proposal["angle"],
            )
            shape64 = shape.double()
            # Gauss-Newton prediction along the shape: decrease ~ slope**2 / curvature,
            # with curvature from the illumination (pseudo-Hessian).
            slope = float((g * shape64).sum())
            curvature = float((h * shape64.square()).sum())
            score = slope * slope / max(curvature, floor)
            records.append(
                dict(
                    region=region,
                    proposal=proposal,
                    **detail,
                    slope=slope,
                    curvature=curvature,
                    curvature_floor=floor,
                    score=score,
                )
            )

        # Stable raster-region tie break. Every region is scored before truncation.
        ranked = sorted(
            records, key=lambda record: (-record["score"], record["region"])
        )
        selected = [r for r in ranked if r["score"] > 0][:max_candidates]
        self.stats = dict(
            regions=len(groups),
            evaluations=len(groups),
            sampled_shapes=len(groups),
            selected=len(selected),
            best_score=ranked[0]["score"] if ranked else 0.0,
            seconds=perf_counter() - start,
        )
        self.rounds.append(
            dict(
                **self.stats,
                candidates=records,
                selected_regions=[r["region"] for r in selected],
            )
        )
        return [dict(r["proposal"]) for r in selected]
