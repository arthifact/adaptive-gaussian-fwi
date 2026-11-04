# Paired synthetic comparison

The experiment uses two rescaled velocity models (Marmousi and Overthrust), two noise seeds (2 and 3), and all four methods at the default iteration budgets. Each model/seed pair uses identical acquisition, observations, velocity bounds, starting ramp and frequency schedule across methods. All methods share the same final three grid bands. Treat the two noise seeds as paired repeats within each model.

## Reproduce the metrics

The `runs/<model>-seed<seed>/` directories contain the numerical archives from the unmodified command-line programs. To recompute `comparison.csv` from the project root:

```sh
uv run --frozen python -m scripts.score_comparison results/runs/*/*.npz --output comparison.csv
```

The scorer rejects duplicate or missing methods, mismatched observations, input hashes, starting grids, reference models, software versions, survey settings, stage maps and iteration schedules. RMSE and relative L2 use the saved velocities in float64. Fitting shot counts are cumulative through each listed band. Reporting forward solves are separate. The number of fitted values is the number of active trainable scalars at that stage. Each adaptive Gaussian has six. The fixed control fits one amplitude per lattice atom.

Training sources occupy horizontal indices 0, 17, 34, 52 and 69. Validation sources occupy 8, 26, 43 and 61. All receivers are on the top row. Validation uses clean synthetic recordings generated with an eighth-order spatial stencil. Fourth-order predictions are scored by the squared norm of the low-pass residual divided by the squared norm of the filtered clean data. This tests interpolation to four unused source locations on the same earth model.

The 20 Hz checkpoint shows the effect of ending grid refinement one band earlier. The `heldout_30_hz` column evaluates every checkpoint at a common 30 Hz cutoff. The `fine_heldout_loss` column repeats the stage-band prediction on a 5 m grid. Bilinear interpolation aligns the 139 × 139 grid with the original 70 × 70 cell centres. Source and receiver positions and absorbing-layer thickness in metres are preserved. Both truth and recovered maps are resampled the same way. `fine_truth_loss` and `truth_heldout_loss` record the stencil discrepancy on each mesh. The fine-grid results are forward sensitivity checks on the recovered maps. The inversion uses 10 m spacing.
