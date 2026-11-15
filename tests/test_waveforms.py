import unittest

import torch

from src.simulation import FWI


class WaveformTests(unittest.TestCase):
    def setUp(self):
        self.survey = dict(
            shape=(10, 12),
            dx=10.0,
            dt=0.001,
            nt=128,
            n_shots=1,
            n_receivers=2,
            source_frequency=15.0,
            bounds=(1000.0, 4000.0),
            surface_velocity=1500.0,
        )
        self.initial = torch.full(self.survey["shape"], 1800.0)

    def test_direct_wave_timing(self):
        # Same physical receivers on the two grids. Compare the extra bandwidth
        # admitted in the last grid stage using the peak near the direct arrival.
        for cells, dx in ((70, 10.0), (139, 5.0)):
            spec = dict(
                self.survey,
                shape=(cells, cells),
                dx=dx,
                n_shots=1,
                n_receivers=70,
                nt=1000,
                bounds=(1000.0, 5500.0),
            )
            problem = FWI(spec)
            self.addCleanup(problem.close)
            speed = 1500.0
            data = problem.forward(
                torch.full(spec["shape"], speed, dtype=torch.float64)
            )[0]
            signal = problem.lowpass(data, 30.0) - problem.lowpass(data, 20.0)
            picks = []
            for receiver in (20, 40):
                offset = float(problem.receivers[0, receiver, 1]) * dx
                arrival = 0.1 + offset / speed
                start, end = (round((arrival + t) / problem.dt) for t in (-0.04, 0.04))
                picks.append(
                    (start + int(signal[receiver, start:end].abs().argmax()))
                    * problem.dt
                )
            self.assertAlmostEqual(
                picks[1] - picks[0], 200.0 / speed, delta=2 * problem.dt
            )
