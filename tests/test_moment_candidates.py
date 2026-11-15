"""Focused arithmetic and native-fitter integration checks; no wave simulator."""

import unittest

import numpy as np
import torch

from src.gaussians import GaussianField
from src.moment_candidates import MomentCandidates, moment_proposal, signed_components


class MomentCandidatesTests(unittest.TestCase):
    def test_bridge_cut(self):
        field = GaussianField(torch.full((70, 70), 2400.0, dtype=torch.float64), 10.0)
        g = torch.zeros_like(field.background)
        g[30:40, 10:20] = -1.0
        g[30:40, 50:60] = -1.0
        candidates = MomentCandidates(field)
        original = candidates.propose(g, torch.ones_like(g), field.background, 6.0)
        g[35, 20:50] = -1e-8
        proposed = candidates.propose(g, torch.ones_like(g), field.background, 6.0)
        self.assertEqual(len(proposed), 2)
        self.assertEqual(sorted(p["x"] for p in proposed), [145.0, 545.0])
        self.assertEqual(
            sorted(original, key=lambda p: p["x"]),
            sorted(proposed, key=lambda p: p["x"]),
        )

    def test_moment_shape(self):
        g = np.zeros((3, 3))
        g[0, 0] = g[0, 2] = 1
        g[2, 2] = 2
        p, d = moment_proposal(
            np.array([[0, 0], [0, 2], [2, 2]]), g, np.ones_like(g), 10.0, 6.0
        )
        self.assertEqual((p["x"], p["z"]), (15.0, 10.0))
        cov = np.array([[75.0, 50.0], [50.0, 100.0]])
        np.testing.assert_allclose(d["covariance_xz_m2"], cov, atol=1e-14)
        eigenvalues = np.linalg.eigvalsh(cov)
        axis = np.array([np.cos(p["angle"]), np.sin(p["angle"])])
        np.testing.assert_allclose(cov @ axis, eigenvalues[-1] * axis, atol=1e-13)
        self.assertAlmostEqual(p["long"], np.sqrt(eigenvalues[-1]))
        self.assertEqual(p["short"], 10.0)
        self.assertEqual(d["nearest_cell"], [1, 2])

    def test_signed_patches(self):
        g = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -2.0], [0.0, 0.0, 0.0]])
        groups = signed_components(g)
        self.assertEqual([len(a) for a in groups], [1, 2, 1])
        self.assertEqual([tuple(a[0]) for a in groups], [(0, 0), (0, 2), (1, 1)])


if __name__ == "__main__":
    unittest.main()
