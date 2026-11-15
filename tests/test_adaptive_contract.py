"""Mathematical derivatives, pruning decisions and preservation of valid state."""

import unittest
from unittest.mock import Mock

import torch

from src.gaussians import GaussianField, gaussian
from src.gaussians_fit import _prune
from src.moment_candidates import MomentCandidates
from src.simulation import FWI


class AdaptiveContractTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_gaussian_derivatives(self):
        field = GaussianField(torch.full((11, 13), 1800.0, dtype=torch.float64), 10.0)
        field.add(z=37.0, x=65.0, long=70.0, short=32.0, angle=0.43, amp=-120.0)
        parameters = field.params.clone().requires_grad_()
        self.assertTrue(
            torch.autograd.gradcheck(
                field.velocity,
                (parameters,),
                eps=1e-5,
                atol=1e-5,
                rtol=1e-4,
                fast_mode=True,
            )
        )

    def test_waveform_gradient(self):
        problem = FWI(
            dict(
                shape=(18, 22),
                dx=10.0,
                dt=0.001,
                nt=256,
                n_shots=2,
                n_receivers=11,
                source_frequency=25.0,
                bounds=(1000.0, 3000.0),
                surface_velocity=1800.0,
            )
        )
        self.addCleanup(problem._pool.shutdown)
        field = GaussianField(
            torch.full(problem.shape, 1800.0, dtype=torch.float64), 10.0
        )
        direction = 30 * gaussian(field.z - 50, field.x - 100, 90.0, 60.0, 0.2)
        problem.observed = problem.forward(field.background + 2 * direction)
        _, gradient = problem.gradient(field.background, 20.0)
        expected = float((gradient * direction).sum())
        step = 1e-3
        difference = (
            problem.loss(field.background + step * direction, 20.0)
            - problem.loss(field.background - step * direction, 20.0)
        ) / (2 * step)
        self.assertGreater(abs(expected), 1e-8)
        self.assertAlmostEqual(difference / expected, 1.0, delta=1e-4)

    def test_ranking_ignores_scaling(self):
        field = GaussianField(torch.full((9, 11), 1800.0, dtype=torch.float64), 10.0)
        g = torch.zeros_like(field.background)
        g[1:4, 2:6] = -torch.arange(1.0, 13.0).reshape(3, 4)
        g[6:8, 7:10] = 2.0
        h = torch.linspace(0.1, 1.0, g.numel(), dtype=g.dtype).reshape(g.shape)
        a, b = MomentCandidates(field), MomentCandidates(field)
        proposals_a = a.propose(g, h, field.background, 9.0)
        proposals_b = b.propose(g * 1e-100, h * 1e100, field.background, 9.0)
        self.assertEqual(
            a.rounds[0]["selected_regions"], b.rounds[0]["selected_regions"]
        )
        for first, second in zip(proposals_a, proposals_b):
            for key in first:
                self.assertAlmostEqual(first[key], second[key], delta=1e-10)
        for first, second in zip(a.rounds[0]["candidates"], b.rounds[0]["candidates"]):
            self.assertAlmostEqual(first["score"], second["score"], delta=1e-10)

    def test_pruning_uses_growth_price(self):
        for threshold, should_remove in ((0.0, False), (1.0, True)):
            with self.subTest(threshold=threshold):
                field = GaussianField(torch.full((5, 7), 1800.0), 10.0)
                for x in (10.0, 50.0):
                    field.add(z=20.0, x=x, long=60.0, short=40.0, angle=0.2, amp=4.0)
                target = field.velocity().clone()
                problem = Mock(bounds=(1000.0, 3000.0))
                problem.loss.side_effect = lambda v, f: float(
                    ((v - target) / 10).square().mean()
                )
                before = field.params.clone()
                keep, measured_loss = _prune(field, problem, 6.0, 0.0, threshold)
                self.assertEqual(measured_loss, problem.loss(field.velocity(), 6.0))
                if should_remove:
                    self.assertEqual(len(field), 0)
                    self.assertFalse(bool(keep.any()))
                    self.assertLessEqual(
                        problem.loss(field.velocity(), 6.0), 2 * threshold
                    )
                else:
                    self.assertTrue(bool(keep.all()))
                    self.assertTrue(torch.equal(field.params, before))


if __name__ == "__main__":
    unittest.main()
