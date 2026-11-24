import unittest
from unittest.mock import Mock, patch

import torch

from src.gaussians import GaussianField, gaussian
from src.gaussians_fit import _grow, fit_gaussians
from src.moment_candidates import MomentCandidates
from src.simulation import FWI


class GaussianGrowthTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_amplitude_keeps_lower_loss(self):
        spec = dict(
            shape=(24, 30),
            dx=10.0,
            dt=0.001,
            nt=400,
            n_shots=2,
            n_receivers=8,
            source_frequency=22.0,
            bounds=(1000.0, 4000.0),
            surface_velocity=2000.0,
        )
        problem = FWI(spec)
        self.addCleanup(problem.close)
        initial = torch.full(spec["shape"], 2000.0, dtype=torch.float64)
        candidate = dict(z=60.0, x=150.0, long=125.0, short=90.0, angle=0.35)
        truth, fitted = GaussianField(initial, 10.0), GaussianField(initial, 10.0)
        truth.add(**candidate, amp=100.0)
        problem.observed = problem.forward(truth.velocity())
        problem.noise_energy = {9: 0.0}
        measure = problem.loss
        measured = []

        def record(grid, frequency):
            measured.append(measure(grid, frequency))
            return measured[-1]

        with (
            patch.object(MomentCandidates, "propose", return_value=[candidate]),
            patch.object(problem, "loss", side_effect=record),
        ):
            (incoming,) = fit_gaussians(fitted, problem, 9, iterations=1)
        probe, estimate = measured
        self.assertEqual(probe, 0.0)
        self.assertLess(probe, estimate)
        self.assertLess(estimate, incoming)
        self.assertEqual(measure(fitted.velocity(), 9), probe)
        self.assertTrue(torch.equal(fitted.params, truth.params))

    def test_pruning_keeps_useful_gaussian(self):
        problem = FWI(
            {
                "shape": (32, 70),
                "dx": 10.0,
                "dt": 0.001,
                "nt": 800,
                "n_shots": 3,
                "n_receivers": 35,
                "source_frequency": 15.0,
                "bounds": (1000.0, 3000.0),
                "surface_velocity": 2000.0,
            }
        )
        self.addCleanup(problem._pool.shutdown)
        initial = torch.full(problem.shape, 2000.0)
        kept = GaussianField(initial, problem.dx)
        truth = initial + 4 * gaussian(kept.z - 140, kept.x - 320, 240.0, 120.0, 0.25)
        problem.observed = problem.forward(truth)
        problem.noise_energy[6] = 0.0

        default = GaussianField(initial, problem.dx)
        # Isolate admission and pruning from the number of signed gradient regions.
        candidate = dict(z=140.0, x=320.0, long=240.0, short=120.0, angle=0.25)
        with patch.object(MomentCandidates, "propose", return_value=[candidate]):
            fit_gaussians(default, problem, 6, iterations=1, prune=True)
            fit_gaussians(kept, problem, 6, iterations=1, prune=False)
        self.assertEqual(len(default), 1)
        self.assertEqual(len(kept), 1)
        self.assertLessEqual(abs(kept.params[0, 5].item()), 5.0)
        torch.testing.assert_close(default.params, kept.params, rtol=0, atol=0)
        self.assertLess(problem.loss(default.velocity(), 6), problem.loss(initial, 6))

    def test_growth_needs_threshold(self):
        initial = torch.full((3, 3), 2000.0)

        def objective(velocity):
            step = (velocity[1, 1] - 2000.0) / 100.0
            return ((1 - step) * (1 + step / 4)).square()

        velocity = initial.clone().requires_grad_()
        loss = objective(velocity)
        gradient = torch.autograd.grad(loss, velocity)[0]
        problem = Mock(bounds=(1000.0, 4000.0))
        problem.loss.side_effect = lambda v, f: float(objective(v))
        candidates = Mock()
        candidates.propose.return_value = [
            dict(z=10.0, x=10.0, long=80.0, short=80.0, angle=0.0)
        ]

        # The 150 m/s estimate gains about 0.53; the 100 m/s trial gains exactly 1.
        for threshold, expected in ((0.5, [100.0]), (1.0, [100.0]), (1.01, [])):
            with self.subTest(threshold=threshold):
                field = GaussianField(initial, 10.0)
                _grow(
                    field,
                    problem,
                    candidates,
                    gradient,
                    torch.ones_like(initial),
                    9,
                    loss.item(),
                    threshold,
                )
                torch.testing.assert_close(
                    field.params[:, 5], initial.new_tensor(expected)
                )
