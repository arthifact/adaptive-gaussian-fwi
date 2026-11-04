import math
import unittest

import torch

from src.simulation import FWI


class WaveformTests(unittest.TestCase):
    def setUp(self):
        self.survey = dict(shape=(10, 12), dx=10.0, dt=0.001, nt=128, n_shots=1, n_receivers=2,
                           source_frequency=15.0, bounds=(1000.0, 4000.0), surface_velocity=1500.0)
        self.initial = torch.full(self.survey["shape"], 1800.0)

    def make_problem(self):
        problem = FWI(self.survey)
        self.addCleanup(problem.close)
        return problem

    def test_zero_source_spectrum_cannot_generate_coloured_noise(self):
        problem = self.make_problem()
        problem.wavelet.zero_()
        shots = torch.ones(1, 2, problem.nt)
        with self.assertRaisesRegex(ValueError, "zero or non-finite noise energy"):
            problem.add_noise(shots, 30., seed=2)

    def test_surface_direct_wave_timing_in_the_final_refinement_band(self):
        # Same physical receivers on the two grids. Compare the extra bandwidth
        # admitted in the last grid stage using the peak near the direct arrival.
        for cells, dx in ((70, 10.), (139, 5.)):
            spec = dict(self.survey, shape=(cells, cells), dx=dx, n_shots=1,
                        n_receivers=70, nt=1000, bounds=(1000., 5500.))
            problem = FWI(spec)
            self.addCleanup(problem.close)
            speed = 1500.
            data = problem.forward(torch.full(spec["shape"], speed, dtype=torch.float64))[0]
            signal = problem.lowpass(data, 30.) - problem.lowpass(data, 20.)
            picks = []
            for receiver in (20, 40):
                offset = float(problem.receivers[0, receiver, 1]) * dx
                arrival = .1 + offset / speed
                start, end = (round((arrival + t) / problem.dt) for t in (-.04, .04))
                picks.append((start + int(signal[receiver, start:end].abs().argmax())) * problem.dt)
            self.assertAlmostEqual(picks[1] - picks[0], 200. / speed, delta=2 * problem.dt)

    def test_source_peak_above_nyquist_is_rejected_before_generating_the_wavelet(self):
        self.survey["source_frequency"] = math.nextafter(500.0, math.inf)
        with self.assertRaisesRegex(ValueError, "source peak frequency.*Nyquist"):
            self.make_problem()

    def test_dc_only_band_rejects_recordings_with_zero_mean(self):
        problem = self.make_problem()
        cutoff = 1 / (2 * problem.nt * problem.dt)
        problem.observed = (torch.arange(problem.nt) % 2 * 2 - 1).float().expand(1, 2, -1)
        with self.assertRaisesRegex(ValueError, "no usable energy"):
            problem.loss(self.initial, cutoff)

    def test_smallest_positive_cutoff_preserves_dc(self):
        problem = self.make_problem()
        ones = torch.ones(1, 2, problem.nt)
        # Zero-padding doubles the record length and halves its mean.
        filtered = problem.lowpass(ones, math.nextafter(0.0, 1.0))
        self.assertTrue(torch.equal(filtered, ones * 0.5))

    def test_double_precision_velocity_works_with_the_default_wavelet(self):
        problem = self.make_problem()
        model = self.initial.double()
        recordings = problem.forward(model, accuracy=4)
        self.assertEqual(recordings.dtype, model.dtype)
        self.assertTrue(torch.isfinite(recordings).all())
        self.assertGreater(float(recordings.norm()), 0.0)

    def test_double_precision_filter_is_independent_of_the_global_default_dtype(self):
        problem = self.make_problem()
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        recordings = torch.randn(1, 2, problem.nt, dtype=torch.float64,
                                 generator=torch.Generator().manual_seed(11))
        torch.set_default_dtype(torch.float32)
        actual = problem.lowpass(recordings, 125.0)
        torch.set_default_dtype(torch.float64)
        self.assertTrue(torch.equal(actual, problem.lowpass(recordings, 125.0)))

    def test_double_precision_noise_estimates_scale_with_recording_amplitude(self):
        self.survey.update(shape=(10, 301), nt=512)  # A far receiver has a pre-arrival window.
        problem = self.make_problem()
        shots = torch.randn(1, 2, problem.nt, dtype=torch.float64,
                            generator=torch.Generator().manual_seed(7))
        scale = 1e-50
        quiet = scale * shots
        noisy = problem.add_noise(quiet, 30)
        self.assertAlmostEqual(float((noisy - quiet).norm() / quiet.norm()), 10**(-30 / 20))
        problem.observed = shots
        expected = problem._noise_variance(6)
        problem.observed = quiet
        self.assertGreater(expected, 0.0)
        self.assertAlmostEqual(problem._noise_variance(6) / scale**2, expected)
