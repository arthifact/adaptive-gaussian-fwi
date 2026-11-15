"""Numerical failures and runner routing, without wave simulations."""

import io
import json
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.gaussians import GaussianField
from src.gaussians_fit import fit_gaussians


class EmptyCandidates:
    def __init__(self, *args):
        pass

    def propose(self, *args, **kwargs):
        return []


class BaselineReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def problem(self, initial):
        problem = Mock(bounds=(1000.0, 5500.0))
        problem.illumination.return_value = torch.ones_like(initial)
        problem.growth_threshold.return_value = 0.0
        return problem

    def test_nan_stops_fit(self):
        initial = torch.full((3, 3), 1800.0)
        for loss, gradient in (
            (float("nan"), torch.ones_like(initial)),
            (1.0, torch.full_like(initial, float("inf"))),
        ):
            with self.subTest(loss=loss):
                field = GaussianField(initial, 10.0)
                before = field.params.clone()
                problem = self.problem(initial)
                problem.gradient.return_value = loss, gradient
                with (
                    patch("src.gaussians_fit.MomentCandidates", EmptyCandidates),
                    self.assertRaisesRegex(FloatingPointError, "6 Hz, update 0"),
                ):
                    fit_gaussians(field, problem, 6, iterations=1)
                self.assertTrue(torch.equal(field.params, before))
                problem.loss.assert_not_called()

    def test_pipeline_saves_all_stages(self):
        true = torch.full((4, 4), 2000.0)
        problem = Mock(
            bounds=(1000.0, 5500.0),
            dx=10.0,
            n_shots=5,
            forward_shots=0,
            adjoint_shots=0,
        )
        problem.forward.return_value = torch.zeros((1, 1, 8))
        problem.add_noise.side_effect = lambda clean, *args: clean + 0.01
        problem.lowpass.side_effect = lambda recordings, f: recordings
        problem.loss.return_value = 1.0
        problem.growth_threshold.return_value = 0.0
        open_file = Path.open

        def open_result(path, mode="r", *args, **kwargs):
            return (
                io.BytesIO() if mode == "wb" else open_file(path, mode, *args, **kwargs)
            )

        def grid_step(pixels, problem, frequency, iterations):
            self.assertEqual(iterations, 350)
            with torch.no_grad():
                pixels.details.add_(frequency)
            return [1.0]

        with (
            patch.object(
                sys, "argv", ["main.py", "--model", "overthrust", "--no-pruning"]
            ),
            patch("src.data_prepare.prepare_model", return_value=true),
            patch("src.simulation.FWI", return_value=problem),
            patch("src.gaussians_fit.fit_gaussians", return_value=[1.0]) as fit,
            patch("src.pixels.fit_pixels", side_effect=grid_step) as refine,
            patch("numpy.savez_compressed") as save,
            patch("paper.make_figures.plot_runs") as plot,
            patch.object(Path, "open", new=open_result),
            patch.object(Path, "replace") as publish,
            patch("pathlib.Path.mkdir"),
            redirect_stdout(io.StringIO()),
        ):
            runpy.run_module("main", run_name="__main__")
        self.assertEqual([c.args[2] for c in fit.call_args_list], [6, 9])
        self.assertTrue(all(c.kwargs["prune"] is False for c in fit.call_args_list))
        settings = json.loads(save.call_args.kwargs["settings"])
        self.assertEqual(settings["method"], "gaussian_background_pixels")
        self.assertEqual([c.args[2] for c in refine.call_args_list], [13, 20, 30])
        self.assertEqual(settings["pixel_frequencies"], [13, 20, 30])
        self.assertEqual(settings["placement"], "signed_region_moments")
        self.assertEqual(settings["noise_seed"], 2)
        self.assertFalse(settings["gaussian_pruning"])
        self.assertEqual(len(settings["gaussian_columns"]), 6)
        self.assertEqual(settings["gaussian_method_settings"]["growth_interval"], 10)
        problem.close.assert_called_once()
        self.assertEqual(
            settings["result_selection"],
            dict(
                gaussians="best_measured_loss_plus_complexity_per_band",
                pixels="last_projected_adam_step",
            ),
        )
        self.assertIn("src/moment_candidates.py", settings["source_sha256"])
        self.assertEqual(save.call_args.kwargs["gaussian_backgrounds"].shape, (2, 4, 4))
        maps = save.call_args.kwargs
        np.testing.assert_array_equal(maps["macro"], maps["gaussian_backgrounds"][-1])
        for grid, delta in zip(maps["pixel_models"], (13, 33, 63), strict=True):
            np.testing.assert_array_equal(grid, maps["macro"] + delta)
        np.testing.assert_array_equal(maps["final"], maps["pixel_models"][-1])
        self.assertEqual(settings["survey"]["dx"], 10.0)
        self.assertEqual(settings["survey"]["bounds"], list(problem.bounds))
        archive = publish.call_args.args[0]
        self.assertEqual(
            archive.parts[-4:],
            ("results", "overthrust-seed2", "adaptive_no_pruning", "result.npz"),
        )
        plot.assert_called_once_with([archive])
