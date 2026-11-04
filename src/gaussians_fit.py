"""Fit one frequency band. The public entry point is fit_gaussians below."""

import math
from typing import NamedTuple

import torch

from .gaussians import GaussianField, gaussian
from .method_settings import (
    DEFAULT_BAND_ITERATIONS, METHOD, PARAMETERS_PER_GAUSSIAN,
    comparison_tolerance, minimum_width,
)
from .moment_candidates import MomentCandidates


def _require_finite(value, context):
    finite = bool(torch.isfinite(value).all()) if isinstance(value, torch.Tensor) else math.isfinite(value)
    if not finite:
        raise FloatingPointError(f"Non-finite {context}; Gaussian fitting stopped")


class _MeasuredBirth(NamedTuple):
    parameters: torch.Tensor
    raw: torch.Tensor
    bounded: torch.Tensor
    loss: float


def _measure_birth(field, problem, frequency, candidate, amplitude, current_bounded, label):
    """Evaluate exactly the model that would be stored, including clipping and rounding."""
    parameters = field.params_with_gaussian(**candidate, amp=amplitude)
    raw = field.velocity(parameters)
    _require_finite(raw, f"Gaussian {label} velocity")
    bounded = raw.clamp(*problem.bounds)
    if torch.equal(bounded, current_bounded):
        return None
    loss = problem.loss(bounded, frequency)
    _require_finite(loss, f"Gaussian {label} loss")
    return _MeasuredBirth(parameters, raw, bounded, loss)


def _earns_component(before, after, threshold, dtype):
    gain = before - after
    return gain >= threshold and gain > comparison_tolerance(before, after, dtype)


def _grow(field, problem, candidates, gradient, illumination, frequency, loss, threshold):
    """Rank once. For each shape, admit its lowest measured loss if the gain pays for the atom."""
    raw = field.velocity()
    lower, upper = problem.bounds
    bounded = raw.clamp(lower, upper)
    active = (raw >= lower) & (raw <= upper)
    proposals = candidates.propose(gradient * active, illumination, bounded, frequency)
    for candidate in proposals:
        shape = gaussian(field.z - candidate["z"], field.x - candidate["x"],
                         candidate["long"], candidate["short"], candidate["angle"])
        iz = round(candidate["z"] / field.dx)
        ix = round(candidate["x"] / field.dx)
        local_speed = float(bounded[iz, ix])
        # The gradient is held for the round. Earlier births can change the clipping mask.
        active = (raw >= lower) & (raw <= upper)
        slope = float((gradient * shape * active).sum())
        _require_finite(slope, "Gaussian amplitude slope")
        amplitude = math.copysign(METHOD.probe_speed_fraction * local_speed, -slope)
        probe = _measure_birth(field, problem, frequency, candidate, amplitude, bounded, "trial")
        if probe is None:
            continue

        curvature = 2 * (probe.loss - loss - amplitude * slope) / amplitude**2
        _require_finite(curvature, "Gaussian amplitude curvature")
        estimate = -slope / curvature if curvature > 0 else amplitude
        cap = METHOD.amplitude_cap_fraction * local_speed
        estimate = max(-cap, min(cap, estimate))
        choice = probe
        if estimate != amplitude:
            trial = _measure_birth(field, problem, frequency, candidate, estimate, bounded, "amplitude-estimate")
            if trial is not None and trial.loss < probe.loss:
                choice = trial
        if _earns_component(loss, choice.loss, threshold, raw.dtype):
            field.params = choice.parameters
            raw, bounded, loss = choice.raw, choice.bounded, choice.loss
    return loss


def _prune(field, problem, frequency, loss, threshold):
    """Remove screened rows only when the measured loss rise fits the saved component charge."""
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
    """Update parameters in place. Each atom keeps its own age and two moments."""
    age, mean, variance = state.unbind(1)
    age += 1
    beta1, beta2 = METHOD.adam_beta1, METHOD.adam_beta2
    mean.mul_(beta1).add_((1 - beta1) * gradient)
    variance.mul_(beta2).add_((1 - beta2) * gradient**2)
    corrected_mean = mean / (1 - beta1**age)
    corrected_variance = variance / (1 - beta2**age)
    rates = parameters.new_tensor(METHOD.adam_rates)
    parameters -= rates * corrected_mean / (corrected_variance.sqrt() + METHOD.adam_epsilon)


def _refine(field, adam, parameters, velocity, cell_gradient, frequency, iteration):
    """Take one ordinary Adam step on the working field and enforce the lagged width floor."""
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


def fit_gaussians(field, problem, frequency, iterations=DEFAULT_BAND_ITERATIONS, *, prune=True, decisions=None):
    """Explore one band and write its best measured model into field.

    Each gradient measures the working model. Between structural rounds, Adam moves
    it freely. Only a measured improvement in loss + threshold * count replaces
    the retained field. The return value contains the exploratory pre-update losses.
    """
    problem.validate_cutoff(frequency)
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if iterations == 0:
        return []
    field.validate_parameters(field.params, f"initial velocity at {frequency:g} Hz")
    threshold = problem.birth_threshold(frequency)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("The Gaussian birth threshold must be finite and nonnegative")
    working = GaussianField(field.background, field.dx)
    working.params = field.params.clone()
    candidates = MomentCandidates(working)
    initial = working.velocity()
    _require_finite(initial, f"initial velocity at {frequency:g} Hz")
    illumination = problem.illumination(initial.clamp(*problem.bounds), frequency)
    _require_finite(illumination, f"illumination at {frequency:g} Hz")
    # The three state rows are age, first moment and second moment.
    adam = working.params.new_zeros((len(working), 3, PARAMETERS_PER_GAUSSIAN))
    retained_score = retained_loss = None

    def retain(loss, iteration, kind):
        nonlocal retained_score, retained_loss
        _require_finite(loss, f"measured loss at {frequency:g} Hz, update {iteration}")
        score = loss + threshold * len(working)
        _require_finite(score, "retention objective")
        accepted = (retained_score is None or retained_score - score >
                    comparison_tolerance(retained_score, score, initial.dtype))
        if accepted:
            field.params = working.params.clone()
            retained_score, retained_loss = score, loss
        if decisions is not None:
            decisions.append(dict(iteration=iteration, kind=kind, loss=loss,
                                  gaussians=len(working), threshold=threshold, accepted=accepted,
                                  retained_loss=retained_loss, retained_gaussians=len(field)))

    losses = []
    for iteration in range(iterations):
        parameters = working.params.clone().requires_grad_()
        raw = working.velocity(parameters)
        _require_finite(raw, f"velocity at {frequency:g} Hz, update {iteration}")
        velocity = raw.clamp(*problem.bounds)
        loss, gradient = problem.gradient(velocity.detach(), frequency)
        _require_finite(gradient, f"cell gradient at {frequency:g} Hz, update {iteration}")
        retain(loss, iteration, "gradient")
        losses.append(loss)
        if iteration % METHOD.growth_interval == 0:
            if problem.birth_threshold(frequency) != threshold:
                raise ValueError("Observations and noise information must stay fixed within a frequency stage")
            before = len(working)
            measured = _grow(working, problem, candidates, gradient, illumination, frequency, loss, threshold)
            new_rows = adam.new_zeros((len(working) - before, *adam.shape[1:]))
            adam = torch.cat((adam, new_rows))
            if prune:
                keep, measured = _prune(working, problem, frequency, measured, threshold)
                adam = adam[keep]
            retain(measured, iteration, "structure")
        else:
            adam = _refine(working, adam, parameters, velocity, gradient, frequency, iteration)

    if (iterations - 1) % METHOD.growth_interval != 0:
        endpoint = working.velocity().clamp(*problem.bounds)
        retain(problem.loss(endpoint, frequency), iterations, "endpoint")
    print(f"Gaussians: {frequency:g} Hz, {len(field)} atoms, retained loss {retained_loss:.3e}")
    return losses
