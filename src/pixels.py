"""A fixed background plus pixel corrections, fitted one frequency at a time."""

import math

import torch


class PixelField:
    def __init__(self, background):
        self.background = background.detach().clone()
        self.details = torch.zeros_like(background, requires_grad=True)

    def velocity(self):
        return self.background + self.details


def fit_pixels(model, problem, frequency, iterations=350):
    """Fit one frequency with fresh Adam state; return the loss before each update."""
    problem.validate_cutoff(frequency)
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    losses = []
    if iterations == 0:
        return losses
    optimizer = torch.optim.Adam([model.details], lr=10.0, eps=1e-20)
    for _ in range(iterations):
        velocity = model.velocity().detach()
        loss, gradient = problem.gradient(velocity, frequency)
        if not math.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite pixel loss or gradient")
        losses.append(loss)
        optimizer.zero_grad()
        model.details.grad = gradient
        optimizer.step()
        with torch.no_grad():
            model.details.clamp_(problem.bounds[0] - model.background, problem.bounds[1] - model.background)
    print(f"Pixels: {frequency:g} Hz, last pre-update loss {loss:.3e}")
    return losses
