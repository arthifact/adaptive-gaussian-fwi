"""Guard the publication table against accidentally mixing experiments or endpoints."""

import unittest

import numpy as np

from paper.match_macro_work import fixed_iterations
from paper.score_comparison import METHODS, compare_pair


class PaperComparisonTests(unittest.TestCase):
    def test_matched_work_budget(self):
        for target in (5, 10, 15, 17, 3250, 5520):
            steps = fixed_iterations(target, 5)
            self.assertLessEqual((2 * steps + 1) * 5, target)
            self.assertGreater((2 * steps + 3) * 5, target)
        with self.assertRaises(ValueError):
            fixed_iterations(4, 5)

    def test_scorer_rejects_unpaired_runs(self):
        truth = np.ones((70, 70), dtype=np.float32)
        survey = dict(shape=[70, 70], n_shots=5, n_receivers=None)
        settings = dict(
            model="marmousi",
            noise_seed=2,
            noise_db=30,
            input_sha256="input",
            observation_sha256="shots",
            versions={},
            survey=survey,
        )
        runs = {m: (settings, [], truth, truth, [], 0) for m in METHODS}
        for other_settings, other_truth in [
            (dict(settings, observation_sha256="different"), truth),
            (settings, truth.astype(np.float64)),
        ]:
            runs[METHODS[1]] = (other_settings, [], other_truth, truth, [], 0)
            with self.assertRaisesRegex(ValueError, "same data and settings"):
                compare_pair(runs)


class StructuralSimilarityTests(unittest.TestCase):
    def test_ssim_identical(self):
        from paper.score_comparison import ssim

        generator = np.random.default_rng(0)
        truth = 1500 + 3000 * generator.random((20, 24))
        model = truth + 200 * generator.standard_normal(truth.shape)
        self.assertAlmostEqual(ssim(truth, truth, 4500.0), 1.0, places=12)
        self.assertAlmostEqual(
            ssim(model, truth, 4500.0), ssim(truth, model, 4500.0), places=12
        )
        self.assertLess(ssim(model, truth, 4500.0), 1.0)

    def test_ssim_flat_maps(self):
        from paper.score_comparison import SSIM_K1, ssim

        a, b, data_range = 2000.0, 2500.0, 4500.0
        c1 = (SSIM_K1 * data_range) ** 2
        expected = (2 * a * b + c1) / (a * a + b * b + c1)
        self.assertAlmostEqual(
            ssim(np.full((15, 15), a), np.full((15, 15), b), data_range),
            expected,
            places=10,
        )
