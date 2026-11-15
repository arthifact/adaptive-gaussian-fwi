import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


class MainTests(unittest.TestCase):
    def test_real_run(self):
        import main

        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        project = Path(main.__file__).resolve().parent
        survey = dict(main.survey, shape=(18, 22), n_shots=2, n_receivers=6, nt=256)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for name in ("main.py", "uv.lock"):
                shutil.copy2(project / name, root / name)
            shutil.copytree(
                project / "src",
                root / "src",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            (root / "paper").mkdir()
            shutil.copy2(project / "paper" / "make_figures.py", root / "paper")
            (root / "data").mkdir()
            # Double-precision input must give the same ramp precision as the controls.
            values = np.load(project / "data" / "marmousi.npy").astype(np.float64)
            model_path = root / "data" / "marmousi.npy"
            np.save(model_path, values)
            with (
                patch.object(main, "__file__", str(root / "main.py")),
                patch.object(main, "survey", survey),
                patch(
                    "src.data_prepare.__file__", str(root / "src" / "data_prepare.py")
                ),
                patch.object(main, "gaussian_iterations", 2),
                patch.object(main, "pixel_iterations", 2),
            ):
                full = main.run_experiment("marmousi", 2)
                partial = main.run_experiment("marmousi", 2, macro_only=True)
            with (
                np.load(full, allow_pickle=False) as result,
                np.load(partial, allow_pickle=False) as macro,
            ):
                from comparisons import fixed_gaussians

                with (
                    patch.object(fixed_gaussians, "ROOT", root),
                    patch.object(fixed_gaussians, "SHAPE", survey["shape"]),
                ):
                    _, control_ramp = fixed_gaussians.load_models("marmousi")
                np.testing.assert_array_equal(
                    result["initial"], control_ramp.numpy(), strict=True
                )
                for name in ("true", "macro", "final"):
                    self.assertEqual(result[name].dtype, np.float64)
                settings = json.loads(result["settings"].item())
                self.assertEqual(
                    settings["input_sha256"],
                    hashlib.sha256(model_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    settings["result_selection"]["pixels"], "last_projected_adam_step"
                )
                np.testing.assert_array_equal(result["gaussian_frequencies"], [6, 9])
                np.testing.assert_array_equal(result["pixel_frequencies"], [13, 20, 30])
                np.testing.assert_array_equal(
                    result["macro"], macro["final"], strict=True
                )
                np.testing.assert_array_equal(
                    result["final"], result["pixel_models"][-1], strict=True
                )
                self.assertEqual(macro["pixel_models"].shape, (0, *survey["shape"]))
                self.assertFalse(np.array_equal(result["macro"], result["final"]))
                self.assertTrue(full.with_name("result.png").is_file())
                self.assertEqual(
                    full.relative_to(root).as_posix(),
                    "results/marmousi-seed2/adaptive/result.npz",
                )
                self.assertEqual(
                    partial.relative_to(root).as_posix(),
                    "results/marmousi-seed2/adaptive_macro/result.npz",
                )
