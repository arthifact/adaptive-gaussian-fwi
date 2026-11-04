"""The independent control files must still solve the same physical experiment."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import unittest

import torch

from comparisons import fixed_gaussians, pixels_plain, pixels_smoothed
from src.simulation import FWI
from src.pixels import PixelField, fit_pixels


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_three_independent_wave_operators_match_the_adaptive_operator(self):
        shape = (12, 17)
        truth = torch.linspace(1800., 2200., shape[0])[:, None].expand(shape).contiguous()
        spec = dict(shape=shape, dx=10., dt=.001, nt=1000, n_shots=5, n_receivers=None,
                    source_frequency=15., surface_velocity=1800., bounds=(1000., 5500.))
        initial = torch.full_like(truth, 1900.)
        with closing(FWI(spec)) as adaptive, ThreadPoolExecutor(5) as pool:
            clean = adaptive.forward(truth, accuracy=8)
            adaptive.observed = adaptive.add_noise(clean, 30., 101)
            adaptive_loss, adaptive_gradient = adaptive.gradient(initial, 9.)
            for module in (fixed_gaussians, pixels_plain, pixels_smoothed):
                with self.subTest(method=module.METHOD_NAME):
                    control = module.Survey(truth, 101, pool)
                    control.start_band(9.)
                    self.assertTrue(torch.equal(adaptive.observed, control.observed))
                    loss, gradient = control.gradient(initial)
                    self.assertEqual(loss, adaptive_loss)
                    self.assertTrue(torch.equal(gradient, adaptive_gradient))
                    self.assertEqual(control.loss(initial), adaptive.loss(initial, 9.))
                    self.assertEqual((control.forwards, control.adjoints), (15, 5))

    def test_fixed_bank_preserves_incoming_field_exactly(self):
        initial = torch.linspace(1500., 3000., 70)[:, None].expand(70, 70).clone()
        field = fixed_gaussians.FixedField(initial)
        field.extend(6)
        self.assertEqual(len(field.amplitudes), 64)
        field.amplitudes.copy_(torch.linspace(-50., 140., 64))
        incoming = field.velocity()
        field.extend(9)
        self.assertEqual(len(field.amplitudes), 208)
        self.assertTrue(torch.equal(field.velocity(), incoming))

    def test_smoothed_pixel_gradient_preserves_constants_and_has_no_preferred_axis(self):
        image = torch.ones(11, 11, dtype=torch.float64)
        torch.testing.assert_close(pixels_smoothed.smooth_gradient(image, 1.3), image)
        image.zero_()
        image[5, 5] = 1.
        filtered = pixels_smoothed.smooth_gradient(image, 1.3)
        self.assertGreater(float(filtered[4, 5]), 0.)
        self.assertLess(float(filtered[5, 5]), 1.)
        torch.testing.assert_close(filtered, filtered.T)

    def test_plain_grid_refinement_matches_the_native_pixel_handoff_on_real_waves(self):
        shape = (12, 17)
        truth = torch.linspace(1750., 2100., shape[0])[:, None].expand(shape).contiguous()
        background = torch.full_like(truth, 1900.)
        spec = dict(shape=shape, dx=10., dt=.001, nt=1000, n_shots=5, n_receivers=None,
                    source_frequency=15., surface_velocity=1750., bounds=(1000., 5500.))
        with closing(FWI(spec)) as native, ThreadPoolExecutor(5) as pool:
            fixed = fixed_gaussians.Survey(truth, 101, pool)
            native.observed = fixed.observed
            field = PixelField(background)
            self.assertTrue(torch.equal(field.velocity(), background))
            grids, records = fixed_gaussians.refine_grid(background, fixed, (13, 20, 30), iterations=2)
            for band, grid, row in zip((13, 20, 30), grids, records, strict=True):
                losses = fit_pixels(field, native, band, iterations=2)
                self.assertEqual(losses, row["loss"])
                self.assertTrue(torch.equal(torch.from_numpy(grid), field.velocity().detach()))
                self.assertTrue(torch.equal(field.background, background))
                self.assertEqual(row["forward_shots"], 10)
                self.assertEqual(row["adjoint_shots"], 10)
                self.assertEqual(row["reporting_forward_shots"], 5)
            self.assertFalse(torch.equal(field.velocity(), background))
