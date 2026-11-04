"""Guard the publication table against accidentally mixing experiments or endpoints."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.tabulate_matched_work import tabulate
from scripts.match_macro_work import fixed_iterations
from scripts.score_comparison import METHODS, compare_pair, score


class PaperComparisonTests(unittest.TestCase):
    def test_work_budget_includes_endpoint_and_never_overspends(self):
        for target in (5, 10, 15, 17, 3250, 5520):
            steps = fixed_iterations(target, 5)
            self.assertLessEqual((2*steps+1)*5, target)
            self.assertGreater((2*steps+3)*5, target)
        with self.assertRaises(ValueError):
            fixed_iterations(4, 5)

    def test_matched_table_rejects_an_incomplete_set_of_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            case = results / "runs" / "marmousi-seed2"
            case.mkdir(parents=True)
            (case / "adaptive.npz").touch()
            with self.assertRaisesRegex(ValueError, "one matched control"):
                tabulate(results)

    def test_duplicate_method_is_rejected_before_solving(self):
        settings = dict(method=METHODS[0], model="marmousi", noise_seed=2)
        with patch("scripts.score_comparison.read_run", return_value=(settings,)):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                score([Path("first.npz"), Path("second.npz")])

    def test_missing_control_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "four methods"):
            compare_pair({METHODS[0]: None})

    def test_matching_seed_cannot_hide_different_shots_or_precision(self):
        truth = np.ones((70, 70), dtype=np.float32)
        survey = dict(shape=[70, 70], n_shots=5, n_receivers=None)
        settings = dict(model="marmousi", noise_seed=2, noise_db=30, input_sha256="input",
                        observation_sha256="shots", versions={}, survey=survey)
        runs = {m: (settings, [], truth, truth, [], 0) for m in METHODS}
        for other_settings, other_truth in [(dict(settings, observation_sha256="different"), truth),
                                             (settings, truth.astype(np.float64))]:
            runs[METHODS[1]] = (other_settings, [], other_truth, truth, [], 0)
            with self.assertRaisesRegex(ValueError, "Unpaired"):
                compare_pair(runs)
