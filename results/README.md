# Paired synthetic comparison

The experiment uses two rescaled velocity models (Marmousi and Overthrust), two noise seeds (2 and 3), and all four methods at the default iteration budgets. Each model/seed pair uses identical acquisition, observations, velocity bounds, starting ramp and frequency schedule across methods. All methods share the same final three grid bands. Treat the two noise seeds as paired repeats within each model.

## Interpretation

The adaptive method returned the lowest full-grid velocity RMSE of the four standard methods at both 9 Hz and 30 Hz in all four model/seed pairs. The starting ramps had RMSE of 862.2 m/s on Marmousi and 2,368.6 m/s on Overthrust. The adaptive 9 Hz error was lower than the work-matched fixed lattice in all four pairs. At seed 2 on Overthrust, matching work reduced the fixed control’s 9 Hz error from 1,026.6 to 399.0 m/s, compared with 369.2 m/s for adaptive. The equal-iteration comparison alone exaggerates the gap on that case.

Held-out waveform rankings depend on the model, frequency and mesh. At 9 Hz on the 10 m grid, plain pixels predicted the unseen Marmousi shots best and adaptive predicted the unseen Overthrust shots best for both seeds. At 30 Hz, smoothed pixels had the lowest held-out loss on Marmousi and fixed Gaussians had the lowest on Overthrust. On the 5 m grid, smoothed pixels had the lowest 30 Hz held-out loss in all four pairs. The final adaptive band improved both velocity RMSE and common-30 Hz held-out loss over its own 20 Hz checkpoint in all four pairs.

These are descriptive results for two rescaled models, two noise realizations and a single acquisition. The Gaussian comparison changes geometry, placement and trainable parameter count together. The standard adaptive runs used 12,445 to 12,650 fitting forward shots, compared with 8,510 for fixed Gaussians. Each used 8,500 adjoint shots. At 9 Hz the adaptive atoms represent 342 to 384 fitted values, compared with 208 fixed amplitudes and 4,900 pixels. Concurrent execution precludes a wall-clock speed claim.

## Reproduce the metrics

The `runs/<model>-seed<seed>/` directories contain the numerical archives from the unmodified command-line programs. Run the following command from the project root to recompute `comparison.csv`.

```sh
uv run --frozen python -m scripts.score_comparison results/runs/*/*.npz --output comparison.csv
```

The scorer rejects duplicate or missing methods, mismatched observations, input hashes, starting grids, reference models, software versions, survey settings, stage maps and iteration schedules. RMSE and relative L2 use the saved velocities in float64. Fitting shot counts are cumulative through each listed band. Reporting forward solves are separate. The number of fitted values is the number of active trainable scalars at that stage. Each adaptive Gaussian has six. The fixed control fits one amplitude per lattice atom.

Training sources occupy horizontal indices 0, 17, 34, 52 and 69. Validation sources occupy 8, 26, 43 and 61. All receivers are on the top row. Validation uses clean synthetic recordings generated with an eighth-order spatial stencil. Fourth-order predictions are scored by the squared norm of the low-pass residual divided by the squared norm of the filtered clean data. This tests interpolation to four unused source locations on the same earth model.

The 20 Hz checkpoint shows the effect of ending grid refinement one band earlier. The `heldout_30_hz` column evaluates every checkpoint at a common 30 Hz cutoff. The `fine_heldout_loss` column repeats the stage-band prediction on a 5 m grid. Bilinear interpolation aligns the 139 × 139 grid with the original 70 × 70 cell centres. Source and receiver positions and absorbing-layer thickness in metres are preserved. Both truth and recovered maps are resampled the same way. `fine_truth_loss` and `truth_heldout_loss` record the stencil discrepancy on each mesh. The fine-grid results are forward sensitivity checks on the recovered maps. The inversion uses 10 m spacing.


## Matched-work macro control

For each adaptive run, a separate fixed-lattice inversion gets the adaptive method’s observed combined fitting forward and adjoint count at each of 6 and 9 Hz. The fixed control spends one extra forward per shot to test its endpoint and one forward plus one adjoint per shot for each update. The script buys the largest integer number of updates within that band’s budget. It can underspend by fewer than ten shot propagations. No RMSE or held-out score enters this allocation.

The lattice, optimizer, initial ramp, seeded observations and retained-state rule are the same as in the standard fixed control. Matching the sum treats each propagation as one unit. Forward and adjoint counts are also reported separately. Different receiver layouts and recording requirements can change the actual cost per shot. The comparison tests low-frequency recovery at comparable nominal work. Both geometry and placement still differ between parameterizations.

```sh
uv run --frozen python -m scripts.match_macro_work results/runs/marmousi-seed2/adaptive.npz --output fixed-matched-marmousi-seed2.npz
```


To rebuild the work-matched CSV from the archived records, use the following command.

```sh
uv run --frozen python -m scripts.tabulate_matched_work results --output matched-work.csv
```

The runs used Python 3.12.13 on macOS arm64, Deepwave 0.0.27, PyTorch 2.14.1 and NumPy 2.5.4. The inversion and comparison runner source is recorded through local commit `bb2c1f8`. Each archive also records file hashes. Raw results below come from the corrected amplitude-selection rule.

## Full-budget results

Velocity RMSE in m/s. Lower is better. The four columns refer to the adaptive Gaussian, fixed Gaussian, plain pixel and smoothed-gradient pixel methods. Each row compares identical seeded observations and the same starting ramp.

| Model | Seed | Band | Adaptive | Fixed | Plain | Smoothed |
|---|---:|---:|---:|---:|---:|---:|
| Marmousi | 2 | 9 Hz | **516.3** | 597.1 | 781.2 | 769.3 |
| Marmousi | 2 | 30 Hz | **469.8** | 539.0 | 699.9 | 612.0 |
| Marmousi | 3 | 9 Hz | **589.1** | 603.7 | 785.8 | 781.9 |
| Marmousi | 3 | 30 Hz | **512.1** | 551.9 | 696.7 | 603.5 |
| Overthrust | 2 | 9 Hz | **369.2** | 1026.6 | 2257.7 | 2092.5 |
| Overthrust | 2 | 30 Hz | **319.0** | 971.6 | 2228.1 | 2074.4 |
| Overthrust | 3 | 9 Hz | **370.7** | 411.6 | 2272.5 | 2098.4 |
| Overthrust | 3 | 30 Hz | **307.2** | 326.3 | 2260.1 | 2115.3 |

Held-out clean waveform loss on the training mesh, multiplied by 1,000 for readability. Lower is better. The scores use the four unseen source locations, with model selection already complete.

| Model | Seed | Band | Adaptive | Fixed | Plain | Smoothed |
|---|---:|---:|---:|---:|---:|---:|
| Marmousi | 2 | 9 Hz | 1.256 | 1.645 | **0.406** | 0.443 |
| Marmousi | 2 | 30 Hz | 1.053 | 1.056 | 1.035 | **0.970** |
| Marmousi | 3 | 9 Hz | 1.466 | 1.624 | **0.267** | 0.436 |
| Marmousi | 3 | 30 Hz | 1.040 | 1.033 | 1.056 | **0.981** |
| Overthrust | 2 | 9 Hz | **0.436** | 0.447 | 6.042 | 3.736 |
| Overthrust | 2 | 30 Hz | 0.361 | **0.351** | 1.160 | 0.513 |
| Overthrust | 3 | 9 Hz | **0.413** | 0.436 | 5.956 | 3.737 |
| Overthrust | 3 | 30 Hz | 0.475 | **0.452** | 1.262 | 0.561 |

### Work and macro model size

| Model | Seed | Adaptive atoms at 9 Hz | Adaptive fitted values | Fixed fitted amplitudes | Adaptive fitting forwards through 30 Hz | Fixed fitting forwards through 30 Hz |
|---|---:|---:|---:|---:|---:|---:|
| Marmousi | 2 | 64 | 384 | 208 | 12,650 | 8,510 |
| Marmousi | 3 | 57 | 342 | 208 | 12,595 | 8,510 |
| Overthrust | 2 | 63 | 378 | 208 | 12,445 | 8,510 |
| Overthrust | 3 | 60 | 360 | 208 | 12,455 | 8,510 |

Each full run uses 8,500 adjoint shots. Both pixel controls use 8,500 fitting forwards. All four methods optimize 4,900 grid values in the 30 Hz band.

### Fixed lattice with matched macro work

| Model | Seed | Adaptive 9 Hz RMSE | Standard fixed 9 Hz RMSE | Work-matched fixed 9 Hz RMSE | Adaptive / matched propagations through 9 Hz |
|---|---:|---:|---:|---:|---:|
| Marmousi | 2 | 516.3 | 597.1 | 647.8 | 10,650 / 10,650 |
| Marmousi | 3 | 589.1 | 603.7 | 640.3 | 10,595 / 10,590 |
| Overthrust | 2 | 369.2 | 1026.6 | 399.0 | 10,445 / 10,440 |
| Overthrust | 3 | 370.7 | 411.6 | 394.6 | 10,455 / 10,450 |

### Sensitivity checks

| Model | Seed | Adaptive 20 Hz RMSE | Adaptive 30 Hz RMSE | 30 Hz held-out loss at 20 / 30 Hz checkpoints (×1,000) | Adaptive 30 Hz held-out loss on 10 / 5 m meshes (×1,000) |
|---|---:|---:|---:|---:|---:|
| Marmousi | 2 | 494.4 | 469.8 | 3.578 / 1.053 | 1.053 / 0.923 |
| Marmousi | 3 | 536.3 | 512.1 | 3.172 / 1.040 | 1.040 / 0.943 |
| Overthrust | 2 | 333.9 | 319.0 | 1.031 / 0.361 | 0.361 / 0.774 |
| Overthrust | 3 | 323.5 | 307.2 | 1.413 / 0.475 | 0.475 / 0.823 |

The numerical records are [comparison.csv](comparison.csv) and [matched-work.csv](matched-work.csv). Exact run archives are in [runs/](runs/) and [work-matched/](work-matched/). [SHA256SUMS](SHA256SUMS) identifies the archived files.
