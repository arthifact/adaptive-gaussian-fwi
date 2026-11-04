"""Retained-model safety without interrupting the exploratory Adam trajectory."""
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
        self.field = GaussianField(torch.full((7, 9), 1800., dtype=torch.float64), 10.)
        self.field.add(z=30., x=40., long=100., short=100., angle=0., amp=100.)
        self.problem = Mock(bounds=(1000., 3000.))
        self.problem.illumination.return_value = torch.ones_like(self.field.background)
        self.problem.birth_threshold.return_value = 0.
        self.states = []
        self.decisions = []

    def objective(self, target):
        def loss(v, f):
            return float((v[3, 4]-target).square())
        def gradient(v, f):
            g = torch.zeros_like(v); g[3, 4] = 2*(v[3, 4]-target)
            return loss(v, f), g
        self.problem.loss.side_effect = loss
        self.problem.gradient.side_effect = gradient

    def step(self, parameters, state, gradient):
        self.states.append(state.clone())
        parameters[:, 5] += 10.
        state.add_(1.)

    def fit(self, iterations):
        with (patch.object(MomentCandidates, 'propose', return_value=[]),
              patch('src.gaussians_fit._adam_step', side_effect=self.step)):
            return fit_gaussians(self.field, self.problem, 6., iterations,
                                 prune=False, decisions=self.decisions)

    def test_best_inner_snapshot_survives_a_worse_endpoint_without_rolling_back_exploration(self):
        self.objective(1911.)
        losses = self.fit(4)
        self.assertEqual(losses, [121., 121., 1., 81.])
        self.assertEqual(float(self.field.params[0, 5]), 110.)
        self.assertEqual(self.problem.gradient.call_count, 4)
        self.assertEqual(self.problem.loss.call_count, 1)
        self.assertEqual([float(s[0, 0, 0]) for s in self.states], [0., 1., 2.])
        self.assertEqual(self.decisions[-1]['loss'], 361.)
        self.assertEqual(self.decisions[-1]['retained_loss'], 1.)

    def test_improving_final_endpoint_is_measured_and_retained(self):
        self.objective(1910.)
        self.fit(2)
        self.assertEqual(float(self.field.params[0, 5]), 110.)
        self.assertEqual(self.decisions[-1]['kind'], 'endpoint')
        self.assertEqual(self.decisions[-1]['retained_loss'], 0.)
        self.problem.loss.assert_called_once()

    def test_unhelpful_exploration_preserves_the_initial_model(self):
        self.objective(1899.)
        before = self.field.params.clone()
        self.fit(4)
        self.assertTrue(torch.equal(self.field.params, before))
        self.assertEqual([d['retained_loss'] for d in self.decisions], [1.]*len(self.decisions))

    def test_roundoff_sized_improvement_does_not_replace_the_retained_model(self):
        self.problem.gradient.return_value = 1., torch.zeros_like(self.field.background)
        self.problem.loss.return_value = 1.-torch.finfo(torch.float64).eps
        before = self.field.params.clone()
        self.fit(2)
        self.assertTrue(torch.equal(self.field.params, before))
        self.assertFalse(self.decisions[-1]['accepted'])

    def test_nonfinite_endpoint_preserves_the_best_measured_snapshot(self):
        self.objective(1910.)
        self.problem.loss.side_effect = lambda v, f: float('nan')
        with self.assertRaisesRegex(FloatingPointError, 'measured loss'):
            self.fit(3)
        self.assertEqual(float(self.field.params[0, 5]), 110.)

    def test_measured_structure_is_ranked_by_loss_plus_complexity(self):
        self.problem.gradient.return_value = 1., torch.zeros_like(self.field.background)
        self.problem.birth_threshold.return_value = .2
        before = self.field.params.clone()
        def grow(working, *args):
            working.add(z=0., x=0., long=100., short=100., angle=0., amp=20.)
            return .9  # Lower loss, but insufficient to pay for one new component.
        with patch('src.gaussians_fit._grow', side_effect=grow):
            self.fit(1)
        self.assertTrue(torch.equal(self.field.params, before))
        self.assertFalse(self.decisions[-1]['accepted'])
        self.problem.loss.assert_not_called()

    def test_endpoint_cost_is_zero_or_one_independently_of_iteration_budget(self):
        self.objective(1900.)
        for budget in (0, 1, 2, 10, 11, 12):
            with self.subTest(budget=budget):
                self.problem.reset_mock()
                self.fit(budget)
                self.assertEqual(self.problem.gradient.call_count, budget)
                self.assertEqual(self.problem.loss.call_count, int(budget > 0 and (budget-1) % 10 != 0))

    def test_full_width_projection_is_measured_before_retention(self):
        old = self.field.velocity()
        wanted = self.field.params.clone()
        wanted[:, 2:4] = (old[3, 4]/24).log()
        target = self.field.velocity(wanted)
        self.problem.gradient.return_value = float((old-target).square().mean()), torch.ones_like(old)
        self.problem.loss.side_effect = lambda v, f: float((v-target).square().mean())
        def narrow(parameters, state, gradient):
            parameters[:, 2:4] = 0.
        self.step = narrow
        self.fit(2)
        torch.testing.assert_close(self.field.params, wanted, rtol=0, atol=0)
        self.assertEqual(self.decisions[-1]['retained_loss'], 0.)

    def test_changed_complexity_charge_is_rejected(self):
        self.objective(1900.)
        self.problem.birth_threshold.side_effect = [0., 0., 1.]
        with self.assertRaisesRegex(ValueError, 'must stay fixed'):
            self.fit(11)

    def test_clean_wave_inversion_has_monotone_retained_objectives(self):
        problem = FWI(dict(shape=(18, 22), dx=10., dt=.001, nt=256, n_shots=2,
                           n_receivers=11, source_frequency=25., bounds=(1500., 3000.), surface_velocity=1800.))
        self.addCleanup(problem._pool.shutdown)
        field = GaussianField(torch.full(problem.shape, 1800.), 10.)
        truth = GaussianField(field.background, 10.)
        truth.add(z=70., x=100., long=90., short=65., angle=.3, amp=90.)
        problem.observed = problem.forward(truth.velocity())
        problem.noise_energy = {20.: 0.}
        decisions = []
        losses = fit_gaussians(field, problem, 20., iterations=14, decisions=decisions)
        endpoint = problem.loss(field.velocity().clamp(*problem.bounds), 20.)
        self.assertLess(endpoint, losses[0])
        objectives = [d['retained_loss']+d['threshold']*d['retained_gaussians'] for d in decisions]
        self.assertTrue(all(a >= b for a, b in zip(objectives, objectives[1:])))
        self.assertAlmostEqual(endpoint/decisions[-1]['retained_loss'], 1., delta=1e-6)


if __name__ == '__main__':
    unittest.main()
