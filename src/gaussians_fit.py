"""Fit Gaussians for one frequency band (fit_gaussians).

Each iteration runs the forward model for the loss and gradient. Every tenth
iteration adds Gaussians where the gradient points (_grow) and drops faded ones
(_prune); the others take one Adam step on all parameters (_refine).

A Gaussian is added only if it lowers the loss by at least the BIC threshold
(FWI.growth_threshold). The band returns the measured model with the lowest
loss + threshold * count.
"""

import math
from typing import NamedTuple

import torch

from .gaussians import GaussianField, gaussian
from .method_settings import (
    DEFAULT_BAND_ITERATIONS,
    METHOD,
    PARAMETERS_PER_GAUSSIAN,
    comparison_tolerance,
    minimum_width,
)
from .moment_candidates import MomentCandidates


def _require_finite(value, context):
    """Raise if `value` contains NaN or inf."""
    finite = (
        bool(torch.isfinite(value).all())
        if isinstance(value, torch.Tensor)
        else math.isfinite(value)
    )
    if not finite:
        raise FloatingPointError(f"Non-finite {context}; Gaussian fitting stopped")


class _MeasuredCandidate(NamedTuple):
    """A measured candidate: its parameter table, raw and clipped velocity, and loss."""

    parameters: torch.Tensor
    raw: torch.Tensor
    bounded: torch.Tensor
    loss: float


def _measure_candidate(
    field, problem, frequency, candidate, amplitude, current_bounded, label
):
    """Loss of the model that would be stored, after clipping."""
    parameters = field.params_with_gaussian(**candidate, amp=amplitude)
    raw = field.velocity(parameters)
    _require_finite(raw, f"Gaussian {label} velocity")
    bounded = raw.clamp(*problem.bounds)
    if torch.equal(bounded, current_bounded):
        return None
    loss = problem.loss(bounded, frequency)
    _require_finite(loss, f"Gaussian {label} loss")
    return _MeasuredCandidate(parameters, raw, bounded, loss)


def _earns_component(before, after, threshold, dtype):
    """True if the loss gain pays the per-Gaussian charge and exceeds roundoff."""
    gain = before - after
    return gain >= threshold and gain > comparison_tolerance(before, after, dtype)


def _grow(
    field, problem, candidates, gradient, illumination, frequency, loss, threshold
):
    """Try the ranked candidates; add each one whose loss gain beats the threshold."""
    raw = field.velocity()
    lower, upper = problem.bounds
    bounded = raw.clamp(lower, upper)
    active = (raw >= lower) & (raw <= upper)
    proposals = candidates.propose(gradient * active, illumination, bounded, frequency)
    for candidate in proposals:
        shape = gaussian(
            field.z - candidate["z"],
            field.x - candidate["x"],
            candidate["long"],
            candidate["short"],
            candidate["angle"],
        )
        iz = round(candidate["z"] / field.dx)
        ix = round(candidate["x"] / field.dx)
        local_speed = float(bounded[iz, ix])
        # The gradient is fixed for the round, but earlier additions can change the
        # clipping mask.
        active = (raw >= lower) & (raw <= upper)
        # Amplitude line search. Slope of the loss at zero amplitude:
        slope = float((gradient * shape * active).sum())
        _require_finite(slope, "Gaussian amplitude slope")
        # Probe at 5% of local speed, downhill.
        amplitude = math.copysign(METHOD.probe_speed_fraction * local_speed, -slope)
        probe = _measure_candidate(
            field, problem, frequency, candidate, amplitude, bounded, "trial"
        )
        if probe is None:
            continue

        # Parabola through loss, slope and probe loss; its minimum, capped at 30%.
        curvature = 2 * (probe.loss - loss - amplitude * slope) / amplitude**2
        _require_finite(curvature, "Gaussian amplitude curvature")
        estimate = -slope / curvature if curvature > 0 else amplitude
        cap = METHOD.amplitude_cap_fraction * local_speed
        estimate = max(-cap, min(cap, estimate))
        # Keep the lower measured loss; admit only if the gain pays the threshold.
        choice = probe
        if estimate != amplitude:
            trial = _measure_candidate(
                field,
                problem,
                frequency,
                candidate,
                estimate,
                bounded,
                "amplitude-estimate",
            )
            if trial is not None and trial.loss < probe.loss:
                choice = trial
        if _earns_component(loss, choice.loss, threshold, raw.dtype):
            field.params = choice.parameters
            raw, bounded, loss = choice.raw, choice.bounded, choice.loss
    return loss


def _prune(field, problem, frequency, loss, threshold):
    """Drop Gaussians weaker than 5 m/s if the loss rise stays within their price.

    The removal is kept only when the loss rises by no more than what the removed
    Gaussians cost (one threshold each), the same price used when adding them.
    """
    keep = field.params[:, -1].abs() > METHOD.prune_amplitude_m_s
    removed = int((~keep).sum())
    if removed:
        proposed = field.velocity(field.params[keep])
        _require_finite(proposed, "pruned velocity")
        new_loss = problem.loss(proposed.clamp(*problem.bounds), frequency)
        _require_finite(new_loss, "pruned loss")
        roundoff = comparison_tolerance(loss, new_loss, proposed.dtype)
        if new_loss - loss <= removed * threshold + roundoff:
            field.keep(keep)
            return keep, new_loss
    return torch.ones_like(keep), loss


def _adam_step(parameters, state, gradient):
    """Update parameters in place. Each Gaussian keeps its own age and two moments."""
    age, mean, variance = state.unbind(1)
    age += 1
    beta1, beta2 = METHOD.adam_beta1, METHOD.adam_beta2
    mean.mul_(beta1).add_((1 - beta1) * gradient)
    variance.mul_(beta2).add_((1 - beta2) * gradient**2)
    corrected_mean = mean / (1 - beta1**age)
    corrected_variance = variance / (1 - beta2**age)
    rates = parameters.new_tensor(METHOD.adam_rates)
    parameters -= (
        rates * corrected_mean / (corrected_variance.sqrt() + METHOD.adam_epsilon)
    )


def _refine(field, adam, parameters, velocity, cell_gradient, frequency, iteration):
    """One Adam step on every Gaussian, then apply the width floor.

    The width floor is a quarter of the local wavelength (see minimum_width), using
    the velocity from before this step ("lagged"), so narrow features the band
    cannot resolve are not created.
    """
    context = f"at {frequency:g} Hz, update {iteration}"
    velocity.backward(cell_gradient)
    _require_finite(parameters.grad, f"Gaussian gradient {context}")
    proposed, next_adam = field.params.clone(), adam.clone()
    _adam_step(proposed, next_adam, parameters.grad)
    _require_finite(next_adam, f"optimizer state {context}")
    _require_finite(proposed, f"Gaussian parameters {context}")
    iz = (proposed[:, 0] / field.dx).round().long().clamp(0, velocity.shape[0] - 1)
    ix = (proposed[:, 1] / field.dx).round().long().clamp(0, velocity.shape[1] - 1)
    floor = minimum_width(velocity.detach()[iz, ix], frequency, field.dx).log()
    proposed[:, 2:4] = torch.maximum(proposed[:, 2:4], floor[:, None])
    field.validate_parameters(proposed, context)
    _require_finite(field.velocity(proposed), f"updated velocity {context}")
    field.params = proposed
    return next_adam


def fit_gaussians(
    field,
    problem,
    frequency,
    iterations=DEFAULT_BAND_ITERATIONS,
    *,
    prune=True,
    decisions=None,
):
    """Explore one band and write its best measured model into field.

    Each gradient measures the working model. Between structural rounds, Adam moves
    it freely. Only a measured improvement in loss + threshold * count replaces
    the best field. The return value contains the exploratory pre-update losses.
    """
    problem.validate_cutoff(frequency)
    if iterations == 0:
        return []
    field.validate_parameters(field.params, f"initial velocity at {frequency:g} Hz")
    threshold = problem.growth_threshold(frequency)
    working = GaussianField(field.background, field.dx)
    working.params = field.params.clone()
    candidates = MomentCandidates(working)
    initial = working.velocity()
    _require_finite(initial, f"initial velocity at {frequency:g} Hz")
    illumination = problem.illumination(initial.clamp(*problem.bounds), frequency)
    _require_finite(illumination, f"illumination at {frequency:g} Hz")
    # The three state rows are age, first moment and second moment.
    adam = working.params.new_zeros((len(working), 3, PARAMETERS_PER_GAUSSIAN))
    best_score = best_loss = None

    def keep_if_best(loss, iteration, kind):
        """Keep the working model if loss + threshold * count is the best so far."""
        nonlocal best_score, best_loss
        _require_finite(loss, f"measured loss at {frequency:g} Hz, update {iteration}")
        score = loss + threshold * len(working)
        _require_finite(score, "retention objective")
        accepted = best_score is None or best_score - score > comparison_tolerance(
            best_score, score, initial.dtype
        )
        if accepted:
            field.params = working.params.clone()
            best_score, best_loss = score, loss
        if decisions is not None:
            decisions.append(
                dict(
                    iteration=iteration,
                    kind=kind,
                    loss=loss,
                    gaussians=len(working),
                    threshold=threshold,
                    accepted=accepted,
                    retained_loss=best_loss,
                    retained_gaussians=len(field),
                )
            )

    losses = []
    for iteration in range(iterations):
        # Loss and gradient of the working model; keep it if it scores best.
        parameters = working.params.clone().requires_grad_()
        raw = working.velocity(parameters)
        _require_finite(raw, f"velocity at {frequency:g} Hz, update {iteration}")
        velocity = raw.clamp(*problem.bounds)
        loss, gradient = problem.gradient(velocity.detach(), frequency)
        _require_finite(
            gradient, f"cell gradient at {frequency:g} Hz, update {iteration}"
        )
        keep_if_best(loss, iteration, "gradient")
        losses.append(loss)
        if iteration % METHOD.growth_interval == 0:
            # Structure round: grow, then prune.
            before = len(working)
            measured = _grow(
                working,
                problem,
                candidates,
                gradient,
                illumination,
                frequency,
                loss,
                threshold,
            )
            new_rows = adam.new_zeros((len(working) - before, *adam.shape[1:]))
            adam = torch.cat((adam, new_rows))
            if prune:
                keep, measured = _prune(
                    working, problem, frequency, measured, threshold
                )
                adam = adam[keep]
            keep_if_best(measured, iteration, "structure")
        else:
            # Adam step on every Gaussian.
            adam = _refine(
                working, adam, parameters, velocity, gradient, frequency, iteration
            )

    # Measure the final Adam step so it can be kept too.
    if (iterations - 1) % METHOD.growth_interval != 0:
        endpoint = working.velocity().clamp(*problem.bounds)
        keep_if_best(problem.loss(endpoint, frequency), iterations, "endpoint")
    print(f"{frequency:g} Hz: {len(field)} Gaussians, best loss {best_loss:.3e}")
    return losses
