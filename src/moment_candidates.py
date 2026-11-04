"""One Gaussian from each signed gradient region; score all, return the best six."""

import math
from time import perf_counter

import numpy as np
import torch

from .gaussians import gaussian
from .method_settings import METHOD, minimum_width

REGION_RELATIVE_FLOOR = METHOD.bridge_fraction


def signed_components(values):
    """Strict-sign four-neighbour components in raster-start order; zeros separate."""
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
                for zz, xx in ((z-1, x), (z+1, x), (z, x-1), (z, x+1)):
                    if (0 <= zz < nz and 0 <= xx < nx and not seen[zz, xx]
                            and values[zz, xx] != 0 and (values[zz, xx] > 0) == sign):
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
    """Absolute-gradient population moments, with the nearest-cell wavelength floor."""
    weights = np.abs(gradient[indices[:, 0], indices[:, 1]])
    mass = float(weights.sum())
    if not math.isfinite(mass) or mass <= 0:
        raise FloatingPointError("Region gradient mass is not representable; cannot determine its centre")
    weights = weights / mass
    points = indices[:, ::-1] * dx  # covariance axes are (x, z)
    mean = (points * weights[:, None]).sum(axis=0)
    offsets = points - mean
    covariance = (offsets * weights[:, None]).T @ offsets
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise FloatingPointError("Non-finite region centre or covariance")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Python round is also used by the waveform admission code.
    iz, ix = round(float(mean[1]) / dx), round(float(mean[0]) / dx)
    minimum = minimum_width(float(velocity[iz, ix]), frequency, dx)
    widths = np.maximum(np.sqrt(np.maximum(eigenvalues[::-1], 0)), minimum)
    angle = math.atan2(eigenvectors[1, -1], eigenvectors[0, -1]) % math.pi
    proposal = dict(z=float(mean[1]), x=float(mean[0]), long=float(widths[0]),
                    short=float(widths[1]), angle=angle)
    if not all(math.isfinite(value) for value in proposal.values()):
        raise FloatingPointError("Gaussian geometry is not representable at this grid spacing or frequency")
    detail = dict(cells=len(indices), sign=int(np.sign(gradient[tuple(indices[0])])),
                  gradient_mass=mass, covariance_xz_m2=covariance.tolist(),
                  eigenvalues_m2=eigenvalues.tolist(), nearest_cell=[iz, ix],
                  minimum_width_m=minimum)
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
    def propose(self, gradient, illumination, velocity, frequency, max_candidates=METHOD.candidates_per_round):
        start = perf_counter()
        if not math.isfinite(float(frequency)) or frequency <= 0:
            raise ValueError("frequency must be finite and positive")
        if type(max_candidates) is not int or max_candidates < 0:
            raise ValueError("max_candidates must be a nonnegative integer")
        for name, value in (("gradient", gradient), ("illumination", illumination),
                            ("velocity", velocity)):
            if tuple(value.shape) != self.shape or value.device != self.z.device:
                raise ValueError(f"{name} must match the field grid and device")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite floating-point values")
        if bool((illumination < 0).any()) or bool((velocity <= 0).any()):
            raise ValueError("illumination must be nonnegative and velocity positive")

        # Out-of-place division preserves caller tensors even if already float64.
        g = gradient.detach().double()
        gmax = float(g.abs().max())
        if gmax:
            g = g / gmax
        h = illumination.detach().double()
        hmax = float(h.max())
        if gmax and not hmax:
            raise ValueError("nonzero gradient requires positive total illumination")
        if hmax:
            h = h / hmax
        values = gradient.detach().double().cpu().numpy()
        current_velocity = velocity.detach().double().cpu().numpy()
        groups = significant_components(values)
        # The smallest resolvable increment at the scale of total normalized illumination.
        floor = math.ulp(float(h.sum()))
        records = []
        for region, indices in enumerate(groups):
            proposal, detail = moment_proposal(indices, values, current_velocity,
                                                self.dx, float(frequency))
            shape = gaussian(self.z-proposal["z"], self.x-proposal["x"],
                             proposal["long"], proposal["short"], proposal["angle"])
            shape64 = shape.double()
            slope = float((g * shape64).sum())
            curvature = float((h * shape64.square()).sum())
            score = slope * slope / max(curvature, floor)
            if not math.isfinite(score) or not bool(torch.isfinite(shape).all()):
                raise FloatingPointError("nonfinite moment Gaussian or score")
            records.append(dict(region=region, proposal=proposal, **detail,
                                slope=slope, curvature=curvature,
                                curvature_floor=floor, score=score))

        # Stable raster-region tie break. Every region is scored before truncation.
        ranked = sorted(records, key=lambda record: (-record["score"], record["region"]))
        selected = [r for r in ranked if r["score"] > 0][:max_candidates]
        self.stats = dict(regions=len(groups), evaluations=len(groups),
                          sampled_shapes=len(groups), selected=len(selected),
                          best_score=ranked[0]["score"] if ranked else 0.0,
                          seconds=perf_counter()-start)
        self.rounds.append(dict(**self.stats, candidates=records,
                                selected_regions=[r["region"] for r in selected]))
        return [dict(r["proposal"]) for r in selected]
