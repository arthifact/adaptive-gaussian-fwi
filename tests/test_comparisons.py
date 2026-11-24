"""The independent control files must still solve the same physical experiment."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import torch

from comparisons import fixed_gaussians, pixels_plain, pixels_smoothed, pixels_tv
from src.pixels import PixelField, fit_pixels
from src.simulation import FWI


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_baselines_match_simulator(self):
        shape = (12, 17)
        truth = (
            torch.linspace(1800.0, 2200.0, shape[0])[:, None].expand(shape).contiguous()
        )
        spec = dict(
            shape=shape,
            dx=10.0,
            dt=0.001,
            nt=1000,
            n_shots=5,
            n_receivers=None,
            source_frequency=15.0,
            surface_velocity=1800.0,
            bounds=(1000.0, 7000.0),
        )
        initial = torch.full_like(truth, 1900.0)
        with closing(FWI(spec)) as adaptive, ThreadPoolExecutor(5) as pool:
            clean = adaptive.observe(truth)
            adaptive.observed = adaptive.add_noise(clean, 20.0, 101)
            adaptive_loss, adaptive_gradient = adaptive.gradient(initial, 9.0)
            for module in (fixed_gaussians, pixels_plain, pixels_smoothed, pixels_tv):
                with self.subTest(method=module.METHOD_NAME):
                    control = module.Survey(truth, 101, pool)
                    control.start_band(9.0)
                    self.assertTrue(torch.equal(adaptive.observed, control.observed))
                    loss, gradient = control.gradient(initial)
                    self.assertEqual(loss, adaptive_loss)
                    self.assertTrue(torch.equal(gradient, adaptive_gradient))
                    self.assertEqual(control.loss(initial), adaptive.loss(initial, 9.0))
                    self.assertEqual((control.forwards, control.adjoints), (10, 5))

    def test_smoothing_is_isotropic(self):
        image = torch.ones(11, 11, dtype=torch.float64)
        torch.testing.assert_close(pixels_smoothed.smooth_gradient(image, 1.3), image)
        image.zero_()
        image[5, 5] = 1.0
        filtered = pixels_smoothed.smooth_gradient(image, 1.3)
        self.assertGreater(float(filtered[4, 5]), 0.0)
        self.assertLess(float(filtered[5, 5]), 1.0)
        torch.testing.assert_close(filtered, filtered.T)

    def test_pixel_stage_matches_baseline(self):
        shape = (12, 17)
        truth = (
            torch.linspace(1750.0, 2100.0, shape[0])[:, None].expand(shape).contiguous()
        )
        background = torch.full_like(truth, 1900.0)
        spec = dict(
            shape=shape,
            dx=10.0,
            dt=0.001,
            nt=1000,
            n_shots=5,
            n_receivers=None,
            source_frequency=15.0,
            surface_velocity=1750.0,
            bounds=(1000.0, 7000.0),
        )
        with closing(FWI(spec)) as native, ThreadPoolExecutor(5) as pool:
            fixed = fixed_gaussians.Survey(truth, 101, pool)
            native.observed = fixed.observed
            field = PixelField(background)
            self.assertTrue(torch.equal(field.velocity(), background))
            grids, records = fixed_gaussians.refine_grid(
                background, fixed, (13, 20, 30), iterations=2
            )
            for band, grid, row in zip((13, 20, 30), grids, records, strict=True):
                losses = fit_pixels(field, native, band, iterations=2)
                self.assertEqual(losses, row["loss"])
                self.assertTrue(
                    torch.equal(torch.from_numpy(grid), field.velocity().detach())
                )
                self.assertTrue(torch.equal(field.background, background))
                self.assertEqual(row["forward_shots"], 10)
                self.assertEqual(row["adjoint_shots"], 10)
                self.assertEqual(row["reporting_forward_shots"], 5)
            self.assertFalse(torch.equal(field.velocity(), background))


class TotalVariationTests(unittest.TestCase):
    def test_tv_gradient(self):
        model = torch.rand(5, 6, dtype=torch.float64) * 500 + 1500
        _, gradient = pixels_tv.tv_gradient(model)
        step = 1e-4
        for z, x in ((0, 0), (2, 3), (4, 5)):
            up, down = model.clone(), model.clone()
            up[z, x] += step
            down[z, x] -= step
            numeric = (
                pixels_tv.total_variation(up) - pixels_tv.total_variation(down)
            ) / (2 * step)
            self.assertAlmostEqual(float(gradient[z, x]), float(numeric), places=6)

    def test_tv_of_step(self):
        step = torch.full((4, 4), 2000.0, dtype=torch.float64)
        step[2:] = 3000.0
        flat_cost = step.numel() * pixels_tv.TV_EPSILON
        expected = (
            flat_cost
            - 4 * pixels_tv.TV_EPSILON
            + 4 * (1000.0**2 + pixels_tv.TV_EPSILON**2) ** 0.5
        )
        self.assertAlmostEqual(
            float(pixels_tv.total_variation(step)), expected, places=8
        )
