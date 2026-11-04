"""Mathematical derivatives, pruning decisions and preservation of valid state."""

import unittest
from unittest.mock import Mock, patch

import torch

from src.gaussians import GaussianField, gaussian
from src.gaussians_fit import _prune, fit_gaussians
from src.moment_candidates import MomentCandidates
from src.simulation import FWI


class AdaptiveContractTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_all_six_gaussian_derivatives_match_finite_differences(self):
        field = GaussianField(torch.full((11, 13), 1800., dtype=torch.float64), 10.)
        field.add(z=37., x=65., long=70., short=32., angle=.43, amp=-120.)
        parameters = field.params.clone().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(field.velocity, (parameters,),
                                                eps=1e-5, atol=1e-5, rtol=1e-4, fast_mode=True))

    def test_waveform_gradient_matches_a_directional_finite_difference(self):
        problem = FWI(dict(shape=(18, 22), dx=10., dt=.001, nt=256,
                           n_shots=2, n_receivers=11, source_frequency=25.,
                           bounds=(1000., 3000.), surface_velocity=1800.))
        self.addCleanup(problem._pool.shutdown)
        field = GaussianField(torch.full(problem.shape, 1800., dtype=torch.float64), 10.)
        direction = 30*gaussian(field.z-50, field.x-100, 90., 60., .2)
        problem.observed = problem.forward(field.background + 2*direction)
        _, gradient = problem.gradient(field.background, 20.)
        expected = float((gradient*direction).sum())
        step = 1e-3
        difference = (problem.loss(field.background + step*direction, 20.)
                      - problem.loss(field.background - step*direction, 20.)) / (2*step)
        self.assertGreater(abs(expected), 1e-8)
        self.assertAlmostEqual(difference/expected, 1., delta=1e-4)

    def test_geometry_and_scores_ignore_global_gradient_and_illumination_scaling(self):
        field = GaussianField(torch.full((9, 11), 1800., dtype=torch.float64), 10.)
        g = torch.zeros_like(field.background)
        g[1:4, 2:6] = -torch.arange(1., 13.).reshape(3, 4)
        g[6:8, 7:10] = 2.
        h = torch.linspace(.1, 1., g.numel(), dtype=g.dtype).reshape(g.shape)
        a, b = MomentCandidates(field), MomentCandidates(field)
        proposals_a = a.propose(g, h, field.background, 9.)
        proposals_b = b.propose(g*1e-100, h*1e100, field.background, 9.)
        self.assertEqual(a.rounds[0]['selected_regions'], b.rounds[0]['selected_regions'])
        for first, second in zip(proposals_a, proposals_b):
            for key in first:
                self.assertAlmostEqual(first[key], second[key], delta=1e-10)
        for first, second in zip(a.rounds[0]['candidates'], b.rounds[0]['candidates']):
            self.assertAlmostEqual(first['score'], second['score'], delta=1e-10)

    def test_pruning_respects_the_same_complexity_charge_as_growth(self):
        for threshold, should_remove in ((0., False), (1., True)):
            with self.subTest(threshold=threshold):
                field = GaussianField(torch.full((5, 7), 1800.), 10.)
                for x in (10., 50.):
                    field.add(z=20., x=x, long=60., short=40., angle=.2, amp=4.)
                target = field.velocity().clone()
                problem = Mock(bounds=(1000., 3000.))
                problem.loss.side_effect = lambda v, f: float(((v-target)/10).square().mean())
                before = field.params.clone()
                keep, measured_loss = _prune(field, problem, 6., 0., threshold)
                self.assertEqual(measured_loss, problem.loss(field.velocity(), 6.))
                if should_remove:
                    self.assertEqual(len(field), 0)
                    self.assertFalse(bool(keep.any()))
                    self.assertLessEqual(problem.loss(field.velocity(), 6.), 2*threshold)
                else:
                    self.assertTrue(bool(keep.all()))
                    self.assertTrue(torch.equal(field.params, before))

    def test_zero_amplitude_is_removable_and_nonfinite_pruning_does_not_change_state(self):
        field = GaussianField(torch.full((5, 7), 1800.), 10.)
        field.add(z=20., x=30., long=60., short=40., angle=.2, amp=0.)
        problem = Mock(bounds=(1000., 3000.))
        problem.loss.return_value = float('nan')
        before = field.params.clone()
        with self.assertRaisesRegex(FloatingPointError, 'pruned loss'):
            _prune(field, problem, 6., 0., 0.)
        self.assertTrue(torch.equal(before, field.params))
        problem.loss.return_value = 0.
        _prune(field, problem, 6., 0., 0.)
        self.assertEqual(len(field), 0)

    def test_invalid_update_keeps_the_last_valid_parameters(self):
        field = GaussianField(torch.full((5, 7), 1800.), 10.)
        for _ in range(2):
            field.add(z=20., x=30., long=60., short=40., angle=.2, amp=100.)
        before = field.params.clone()
        problem = Mock(bounds=(1000., 3000.))
        problem.illumination.return_value = torch.ones_like(field.background)
        problem.gradient.return_value = 1., torch.ones_like(field.background)
        problem.birth_threshold.return_value = 0.

        def overflow(parameters, state, gradient):
            parameters[:, 5] = torch.finfo(parameters.dtype).max

        with (patch.object(MomentCandidates, 'propose', return_value=[]),
              patch('src.gaussians_fit._adam_step', side_effect=overflow),
              self.assertRaisesRegex(FloatingPointError, 'updated velocity')):
            fit_gaussians(field, problem, 6., iterations=2)
        self.assertTrue(torch.equal(field.params, before))

    def test_invalid_width_that_renders_a_finite_grid_fails_before_wave_work(self):
        field = GaussianField(torch.full((7, 9), 1800., dtype=torch.float64), 10.)
        field.add(z=30., x=25., long=100., short=100., angle=0., amp=100.)
        problem = Mock()
        for log_width in (-1000., 1000.):
            with self.subTest(log_width=log_width):
                field.params[0, 2] = log_width
                self.assertTrue(bool(torch.isfinite(field.velocity()).all()))
                with self.assertRaisesRegex(FloatingPointError, 'widths.*representable'):
                    fit_gaussians(field, problem, 6., iterations=2)
        problem.illumination.assert_not_called()
        problem.gradient.assert_not_called()
        problem.loss.assert_not_called()

    def test_invalid_grid_and_iteration_budget_fail_before_wave_work(self):
        for dx in (0., -1., float('nan'), float('inf')):
            with self.subTest(dx=dx), self.assertRaises(ValueError):
                GaussianField(torch.ones(2, 2), dx)
        for background in (torch.empty(0, 2), torch.full((2, 2), float('nan'))):
            with self.assertRaises(ValueError):
                GaussianField(background, 10.)
        field = GaussianField(torch.ones(2, 2), 10.)
        problem = Mock()
        for budget in (-1, .5, True):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, 'integer'):
                fit_gaussians(field, problem, 6., iterations=budget)
        problem.illumination.assert_not_called()
        problem.gradient.assert_not_called()


if __name__ == '__main__':
    unittest.main()
