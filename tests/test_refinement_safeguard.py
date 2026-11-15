"""Best-model safety without interrupting the exploratory Adam trajectory."""

import unittest
from unittest.mock import Mock, patch

import torch

from src.gaussians import GaussianField
from src.gaussians_fit import fit_gaussians
from src.moment_candidates import MomentCandidates
from src.simulation import FWI


class RefinementSafeguardTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.field = GaussianField(
            torch.full((7, 9), 1800.0, dtype=torch.float64), 10.0
        )
        self.field.add(z=30.0, x=40.0, long=100.0, short=100.0, angle=0.0, amp=100.0)
        self.problem = Mock(bounds=(1000.0, 3000.0))
        self.problem.illumination.return_value = torch.ones_like(self.field.background)
        self.problem.growth_threshold.return_value = 0.0
        self.states = []
        self.decisions = []

    def objective(self, target):
        def loss(v, f):
            return float((v[3, 4] - target).square())

        def gradient(v, f):
            g = torch.zeros_like(v)
            g[3, 4] = 2 * (v[3, 4] - target)
            return loss(v, f), g

        self.problem.loss.side_effect = loss
        self.problem.gradient.side_effect = gradient

    def step(self, parameters, state, gradient):
        self.states.append(state.clone())
        parameters[:, 5] += 10.0
        state.add_(1.0)

    def fit(self, iterations):
        with (
            patch.object(MomentCandidates, "propose", return_value=[]),
            patch("src.gaussians_fit._adam_step", side_effect=self.step),
        ):
            return fit_gaussians(
                self.field,
                self.problem,
                6.0,
                iterations,
                prune=False,
                decisions=self.decisions,
            )

    def test_best_model_kept(self):
        self.objective(1911.0)
        losses = self.fit(4)
        self.assertEqual(losses, [121.0, 121.0, 1.0, 81.0])
        self.assertEqual(float(self.field.params[0, 5]), 110.0)
        self.assertEqual(self.problem.gradient.call_count, 4)
        self.assertEqual(self.problem.loss.call_count, 1)
        self.assertEqual([float(s[0, 0, 0]) for s in self.states], [0.0, 1.0, 2.0])
        self.assertEqual(self.decisions[-1]["loss"], 361.0)
        self.assertEqual(self.decisions[-1]["retained_loss"], 1.0)

    def test_selection_loss_plus_price(self):
        self.problem.gradient.return_value = (
            1.0,
            torch.zeros_like(self.field.background),
        )
        self.problem.growth_threshold.return_value = 0.2
        before = self.field.params.clone()

        def grow(working, *args):
            working.add(z=0.0, x=0.0, long=100.0, short=100.0, angle=0.0, amp=20.0)
            return 0.9  # Lower loss, but insufficient to pay for one new component.

        with patch("src.gaussians_fit._grow", side_effect=grow):
            self.fit(1)
        self.assertTrue(torch.equal(self.field.params, before))
        self.assertFalse(self.decisions[-1]["accepted"])
        self.problem.loss.assert_not_called()

    def test_best_objective_monotone(self):
        problem = FWI(
            dict(
                shape=(18, 22),
                dx=10.0,
                dt=0.001,
                nt=256,
                n_shots=2,
                n_receivers=11,
                source_frequency=25.0,
                bounds=(1500.0, 3000.0),
                surface_velocity=1800.0,
            )
        )
        self.addCleanup(problem._pool.shutdown)
        field = GaussianField(torch.full(problem.shape, 1800.0), 10.0)
        truth = GaussianField(field.background, 10.0)
        truth.add(z=70.0, x=100.0, long=90.0, short=65.0, angle=0.3, amp=90.0)
        problem.observed = problem.forward(truth.velocity())
        problem.noise_energy = {20.0: 0.0}
        decisions = []
        losses = fit_gaussians(field, problem, 20.0, iterations=14, decisions=decisions)
        endpoint = problem.loss(field.velocity().clamp(*problem.bounds), 20.0)
        self.assertLess(endpoint, losses[0])
        objectives = [
            d["retained_loss"] + d["threshold"] * d["retained_gaussians"]
            for d in decisions
        ]
        self.assertTrue(all(a >= b for a, b in zip(objectives, objectives[1:])))
        self.assertAlmostEqual(
            endpoint / decisions[-1]["retained_loss"], 1.0, delta=1e-6
        )


if __name__ == "__main__":
    unittest.main()
