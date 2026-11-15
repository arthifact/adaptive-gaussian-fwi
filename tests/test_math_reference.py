"""Independent checks of the filter, the wave derivative and per-row Adam."""

import unittest

import numpy as np
import torch

from src.gaussians import GaussianField
from src.gaussians_fit import _adam_step
from src.method_settings import METHOD
from src.simulation import FWI


class MathReferenceTests(unittest.TestCase):
    def make_problem(self, nt):
        problem = FWI(
            dict(
                shape=(17, 23),
                dx=10.0,
                dt=0.001,
                nt=nt,
                n_shots=2,
                n_receivers=7,
                source_frequency=25.0,
                bounds=(1000.0, 2800.0),
                surface_velocity=1800.0,
            )
        )
        self.addCleanup(problem.close)
        return problem

    def test_lowpass_spectrum(self):
        generator = np.random.default_rng(42)
        for nt in (65, 80):
            with self.subTest(nt=nt):
                problem = self.make_problem(nt)
                x, y = torch.from_numpy(generator.normal(size=(2, nt)))
                cutoff = 80.0
                nfft = 2 * nt
                frequencies = np.abs(np.fft.fftfreq(nfft, problem.dt))
                transfer = np.ones_like(frequencies)
                transition = (frequencies > 0.7 * cutoff) & (frequencies < cutoff)
                transfer[frequencies >= cutoff] = 0.0
                transfer[transition] = (
                    np.cos(
                        np.pi
                        * (frequencies[transition] - 0.7 * cutoff)
                        / (0.6 * cutoff)
                    )
                    ** 2
                )
                reference = np.fft.ifft(np.fft.fft(x.numpy(), nfft) * transfer).real[
                    :nt
                ]
                filtered_x = problem.lowpass(x, cutoff)
                np.testing.assert_allclose(
                    filtered_x.numpy(), reference, atol=1e-13, rtol=1e-13
                )
                torch.testing.assert_close(
                    x @ problem.lowpass(y, cutoff),
                    filtered_x @ y,
                    atol=1e-13,
                    rtol=1e-13,
                )
                problem.observed = x.expand(2, 7, nt).clone()
                self.assertAlmostEqual(
                    float(problem._band_power_weights(cutoff).sum()),
                    float(np.square(transfer).sum()),
                    places=12,
                )

    def test_wave_loss_through_gaussian(self):
        problem = self.make_problem(256)
        field = GaussianField(
            torch.full(problem.shape, 1800.0, dtype=torch.float64), problem.dx
        )
        field.add(z=65.0, x=100.0, long=64.0, short=33.0, angle=0.45, amp=1120.0)
        theta = field.params.clone().requires_grad_()
        raw = field.velocity(theta)
        self.assertTrue(
            bool((raw > problem.bounds[1]).any() and (raw < problem.bounds[1]).any())
        )
        model = raw.clamp(*problem.bounds)
        problem.observed = problem.forward(field.background, accuracy=8)
        loss, cell_gradient = problem.gradient(model.detach(), 20.0)
        self.assertAlmostEqual(loss, problem.loss(model.detach(), 20.0), places=12)
        (param_gradient,) = torch.autograd.grad(
            model, theta, grad_outputs=cell_gradient
        )
        direction = theta.new_tensor([[2.0, -3.0, 0.05, -0.03, 0.06, 24.0]])
        derivative = float((param_gradient * direction).sum())
        self.assertGreater(abs(derivative), 1e-7)
        for step in (0.01, 0.005):
            plus = field.velocity(field.params + step * direction)
            minus = field.velocity(field.params - step * direction)
            for grid in (plus, minus):
                self.assertTrue(
                    torch.equal(grid > problem.bounds[1], raw > problem.bounds[1])
                )
            measured = (
                problem.loss(plus.clamp(*problem.bounds), 20.0)
                - problem.loss(minus.clamp(*problem.bounds), 20.0)
            ) / (2 * step)
            self.assertAlmostEqual(measured / derivative, 1.0, delta=2e-4)

    def test_adam_matches_pytorch(self):
        theta = torch.arange(6, dtype=torch.float64)[None].clone()
        history = theta.new_zeros(1, 3, 6)
        # Separate parameters give each Gaussian its own Adam step count.
        rows = [
            [torch.nn.Parameter(value.clone()) for value in theta[0]] for _ in range(2)
        ]
        optimizer = torch.optim.Adam(
            [
                dict(params=[row[col] for row in rows], lr=rate)
                for col, rate in enumerate(METHOD.adam_rates)
            ],
            betas=(METHOD.adam_beta1, METHOD.adam_beta2),
            eps=METHOD.adam_epsilon,
            foreach=False,
        )
        pattern = theta.new_tensor([-0.4, 0.2, 0.03, -0.01, 0.06, 0.8])
        for step in range(5):
            if step == 2:
                theta = torch.cat((theta, torch.arange(6, dtype=theta.dtype)[None]))
                history = torch.cat((history, history.new_zeros(1, 3, 6)))
            gradient = torch.stack(
                [torch.roll(pattern, row + step) for row in range(len(theta))]
            )
            _adam_step(theta, history, gradient)
            for row in range(len(theta)):
                for col, parameter in enumerate(rows[row]):
                    parameter.grad = gradient[row, col].clone()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            expected = torch.stack(
                [torch.stack(row) for row in rows[: len(theta)]]
            ).detach()
            torch.testing.assert_close(theta, expected, rtol=1e-12, atol=1e-12)
