import math
"""Focused arithmetic and native-fitter integration checks; no wave simulator."""

import unittest

import numpy as np
import torch

from src.gaussians import GaussianField, gaussian
from src.moment_candidates import MomentCandidates
from src.moment_candidates import moment_proposal, signed_components, significant_components

class MomentCandidatesTests(unittest.TestCase):
    def test_weak_bridge_does_not_merge_the_two_proposals(self):
        field=GaussianField(torch.full((70,70),2400.,dtype=torch.float64),10.)
        g=torch.zeros_like(field.background)
        g[30:40,10:20]=-1.;g[30:40,50:60]=-1.
        candidates=MomentCandidates(field)
        original=candidates.propose(g,torch.ones_like(g),field.background,6.)
        g[35,20:50]=-1e-8
        proposed=candidates.propose(g,torch.ones_like(g),field.background,6.)
        self.assertEqual(len(proposed),2)
        self.assertEqual(sorted(p['x'] for p in proposed),[145.,545.])
        self.assertEqual(sorted(original,key=lambda p:p['x']),sorted(proposed,key=lambda p:p['x']))

    def test_cutoff_is_local_and_equality_is_excluded(self):
        # A separate weak component survives; the bridge at exactly 1% is cut.
        g=np.array([[1.,.01,1.,0.,1e-12,1e-12],[-1.,-.009,-1.,0.,0.,0.]])
        groups=significant_components(g)
        self.assertEqual([a.tolist() for a in groups],
                         [[[0,0]],[[0,2]],[[0,4],[0,5]],[[1,0]],[[1,2]]])
        # A meaningful bridge above the declared threshold is retained.
        g[0,1]=.0101
        self.assertEqual(len(significant_components(g)[0]),3)

    def test_bridge_filter_is_scale_and_sign_invariant_without_mutation(self):
        g=np.array([[1.,1e-8,1.],[0.,0.,0.],[-.1,-.001,-.1]])
        before=g.copy()
        expected=[a.tolist() for a in significant_components(g)]
        for scale in (1e-100,1e100,-1.):
            self.assertEqual([a.tolist() for a in significant_components(g*scale)],expected)
        np.testing.assert_array_equal(g,before)

    def test_weighted_centroid_covariance_and_angle(self):
        g = np.zeros((3, 3)); g[0, 0] = g[0, 2] = 1; g[2, 2] = 2
        p, d = moment_proposal(np.array([[0, 0], [0, 2], [2, 2]]), g,
                               np.ones_like(g), 10., 6.)
        self.assertEqual((p['x'], p['z']), (15., 10.))
        cov = np.array([[75., 50.], [50., 100.]])
        np.testing.assert_allclose(d['covariance_xz_m2'], cov, atol=1e-14)
        eigenvalues = np.linalg.eigvalsh(cov)
        axis = np.array([np.cos(p['angle']), np.sin(p['angle'])])
        np.testing.assert_allclose(cov @ axis, eigenvalues[-1]*axis, atol=1e-13)
        self.assertAlmostEqual(p['long'], np.sqrt(eigenvalues[-1]))
        self.assertEqual(p['short'], 10.)
        self.assertEqual(d['nearest_cell'], [1, 2])

    def test_components_four_neighbours_signs_and_zeros(self):
        g = np.array([[1., 0., -1.], [0., 1., -2.], [0., 0., 0.]])
        groups = signed_components(g)
        self.assertEqual([len(a) for a in groups], [1, 2, 1])
        self.assertEqual([tuple(a[0]) for a in groups], [(0, 0), (0, 2), (1, 1)])

    def test_python_round_current_velocity_floor_single_cell_and_no_width_cap(self):
        g = np.array([[1., 1., 0., 0.]])
        v = np.array([[400., 800., 1200., 1600.]])
        p, d = moment_proposal(np.array([[0, 0], [0, 1]]), g, v, 10., 1.)
        self.assertEqual(d['nearest_cell'], [0, 0])  # round(0.5)==0
        self.assertEqual((p['long'], p['short']), (100., 100.))
        g = np.array([[0., 1., 1., 0.]])
        p, d = moment_proposal(np.array([[0, 1], [0, 2]]), g, v, 10., 1.)
        self.assertEqual(d['nearest_cell'], [0, 2])  # round(1.5)==2
        self.assertEqual((p['long'], p['short']), (300., 300.))
        # The physical floor may exceed the domain diagonal; native code permits it.
        p, d = moment_proposal(np.array([[0, 2]]), g, v, 10., 1.)
        self.assertEqual((p['long'], p['short']), (300., 300.))
        self.assertEqual(d['covariance_xz_m2'], [[0., 0.], [0., 0.]])

    def test_field_dtype_render_score_no_grad_and_no_mutation(self):
        f = GaussianField(torch.full((9, 11), 1800.), 10.)
        f.add(z=-20., x=150., long=200., short=100., angle=.4, amp=10.)
        before = f.params.clone()
        g = torch.zeros_like(f.background, dtype=torch.float64)
        g[2:5, 3:7] = -torch.arange(1., 13.).reshape(3, 4)
        h = torch.linspace(.1, 1., g.numel()).reshape(g.shape).double()
        v = f.velocity().clamp(1000, 5500)
        copies = [a.clone() for a in (g, h, v)]
        c = MomentCandidates(f)
        with torch.no_grad():
            proposals = c.propose(g, h, v, 6.)
        record = c.rounds[0]['candidates'][0]; p = proposals[0]
        shape = gaussian(f.z-p['z'], f.x-p['x'], p['long'], p['short'], p['angle'])
        self.assertEqual(shape.dtype, torch.float32)
        gd, hd = g/g.abs().max(), h/h.max()
        s = float((gd*shape.double()).sum())
        denominator = max(float((hd*shape.double().square()).sum()), math.ulp(float(hd.sum())))
        self.assertEqual(record['score'], s*s/denominator)
        for actual, old in zip((g, h, v), copies):
            self.assertTrue(torch.equal(actual, old))
        self.assertTrue(torch.equal(f.params, before))
        self.assertEqual(set(p), {'z', 'x', 'long', 'short', 'angle'})
        # Admission deliberately keeps the old log-width storage and its roundoff.
        stored = f.params_with_gaussian(**p, amp=20.)[-1]
        self.assertEqual(stored.dtype, torch.float32)
        self.assertEqual(float(stored[2]), float(torch.tensor(np.log(p['long']), dtype=torch.float32)))
        proposals[0]['x'] = -1
        self.assertNotEqual(record['proposal']['x'], -1)

    def test_all_regions_scored_before_budget_no_nms_and_deterministic_ties(self):
        f = GaussianField(torch.ones((31, 31)), 1.)
        g = torch.zeros_like(f.background)
        for x in (2, 6, 10, 14, 18, 22):
            g[15, x] = 1.
        g[15, 28] = .9
        h = torch.full_like(g, 1e-6); h[:, :25] = 1.
        c = MomentCandidates(f)
        c.propose(g, h, f.background, 1.)
        self.assertEqual(c.stats['regions'], 7)
        self.assertEqual(c.stats['evaluations'], 7)
        self.assertEqual(c.stats['selected'], 6)
        self.assertEqual(c.rounds[0]['selected_regions'][0], 6)
        # Broad, highly overlapping proposals both survive; there is no spacing rule.
        f = GaussianField(torch.full((1, 3), 400.), 1.)
        g = torch.tensor([[1., 0., 1.]])
        c = MomentCandidates(f)
        proposals = c.propose(g, torch.ones_like(g), f.background, 1.)
        self.assertEqual(len(proposals), 2)
        self.assertEqual(c.rounds[0]['candidates'][0]['score'], c.rounds[0]['candidates'][1]['score'])
        self.assertEqual(c.rounds[0]['selected_regions'], [0, 1])

    def test_shape_with_zero_illuminated_overlap_has_a_finite_score(self):
        field = GaussianField(torch.full((3, 128), 1800.), 10.)
        gradient = torch.zeros_like(field.background)
        gradient[1, 0] = -1.
        illumination = torch.zeros_like(gradient)
        illumination[1, -1] = 1.
        candidates = MomentCandidates(field)
        proposals = candidates.propose(gradient, illumination, field.background, 100.)
        record = candidates.rounds[0]['candidates'][0]
        self.assertEqual(len(proposals), 1)
        self.assertEqual(record['curvature'], 0.)
        self.assertEqual(record['curvature_floor'], math.ulp(1.))
        self.assertTrue(math.isfinite(record['score']))

    def test_empty_and_invalid_inputs(self):
        f = GaussianField(torch.ones((3, 3)), 1.)
        c = MomentCandidates(f)
        zeros = torch.zeros_like(f.background)
        self.assertEqual(c.propose(zeros, zeros, f.background, 1.), [])
        self.assertEqual(c.stats['evaluations'], 0)
        with self.assertRaisesRegex(ValueError, 'positive total illumination'):
            c.propose(torch.ones_like(zeros), zeros, f.background, 1.)
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            c.propose(zeros, -torch.ones_like(zeros), f.background, 1.)
        with self.assertRaisesRegex(ValueError, 'finite'):
            c.propose(torch.full_like(zeros, float('nan')), zeros, f.background, 1.)


if __name__ == "__main__":
    unittest.main()
