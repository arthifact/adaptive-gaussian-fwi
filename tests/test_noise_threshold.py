"""Regression checks for known, estimated and unavailable observation noise."""

import io
import json
import unittest

import numpy as np
import torch

from src.simulation import FWI


class NoiseThresholdTests(unittest.TestCase):
    def make_problem(self, nx=16, surface_velocity=2000.0, dt=0.001):
        problem = FWI(
            {
                "shape": (16, nx),
                "dx": 10.0,
                "dt": dt,
                "nt": 1000,
                "n_shots": 2,
                "n_receivers": 4,
                "source_frequency": 15.0,
                "bounds": (1000.0, 6000.0),
                "surface_velocity": surface_velocity,
            }
        )
        self.addCleanup(problem._pool.shutdown)
        problem.observed = torch.randn(
            2,
            4,
            1000,
            generator=torch.Generator().manual_seed(0),
        )
        return problem

    def test_clean_data_zero_threshold(self):
        problem = self.make_problem()
        problem.noise_energy[6] = 0.0
        self.assertEqual(problem.growth_threshold(6), 0.0)

    def test_threshold_scale(self):
        problem = self.make_problem()
        problem.noise_energy[6] = 1.0
        self.assertEqual(problem.threshold_scale, 1.0)
        published = problem.growth_threshold(6)
        problem.threshold_scale = 2.0
        self.assertEqual(problem.growth_threshold(6), 2.0 * published)
        problem.threshold_scale = 0.5
        self.assertEqual(problem.growth_threshold(6), 0.5 * published)

    def test_supplied_noise_energy(self):
        problem = self.make_problem(nx=241, surface_velocity=1500.0)
        self.assertGreater(problem.growth_threshold(6), 0.0)
        problem.noise_energy[6] = 0.0
        self.assertEqual(problem.growth_threshold(6), 0.0)

    def test_threshold_from_saved_run(self):
        for nx in (16, 241):  # Without and with a usable pre-arrival window.
            with self.subTest(nx=nx):
                problem = self.make_problem(nx=nx, surface_velocity=1500.0)
                problem.noise_energy = {6: 0.0, 6.5: 1.0, 9.0: 2.0}
                expected = {
                    f: problem.growth_threshold(f) for f in problem.noise_energy
                }
                # Use the same JSON-in-NPZ format as main.py.
                with io.BytesIO() as file:
                    np.savez_compressed(
                        file,
                        settings=json.dumps({"noise_energy": problem.noise_energy}),
                    )
                    file.seek(0)
                    with np.load(file, allow_pickle=False) as archive:
                        saved = json.loads(archive["settings"].item())
                replay = self.make_problem(nx=nx, surface_velocity=1500.0)
                replay.observed = problem.observed.clone()
                replay.noise_energy = saved["noise_energy"]
                for f, threshold in expected.items():
                    self.assertEqual(replay.growth_threshold(f), threshold)


if __name__ == "__main__":
    unittest.main()
