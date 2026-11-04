# Adaptive Gaussian FWI

This synthetic 2-D acoustic experiment fits an adaptive Gaussian velocity background at 6 and 9 Hz, freezes the retained 9 Hz map, then fits one additive velocity correction per grid cell at 13, 20 and 30 Hz. The reported comparisons use spatially rescaled versions of Marmousi and Overthrust.

## Run

Use Python 3.12 and uv from the project root.

```sh
uv sync --frozen
uv run --frozen python -m unittest discover -q
uv run --frozen python main.py --model marmousi --noise-seed 2
```

`--macro-only` stops after 9 Hz. The default enables loss-checked pruning. Use `--no-pruning` for the matching ablation. The acquisition and band schedule are in [main.py](main.py).

## Survey and model scale

The loader area-resamples the **whole** input array to 70 depth cells by 70 horizontal cells. Both computational axes have 10 m spacing and span 690 m between cell centres. Original Marmousi spans 9.2 km horizontally and 3 km in depth. This transformation changes dips, feature sizes and their relation to wavelength. The synthetic recordings, RMSE calculations and plotted metre coordinates refer to this new square domain. Earth-model sources, units, slice definitions, notices and checksums are in [data/README.md](data/README.md). [Original Marmousi dimensions, Figure 1](https://mcee.ou.edu/aaspi/publications/2006/martin_etal_TLE2006.pdf)

Five sources and 70 receivers lie on the top model row. The simulation uses absorbing boundaries on all sides, a 15 Hz Ricker source, a 1 ms sample interval and 1,000 time samples. Clean observations use an eighth-order spatial stencil. Predictions use fourth order. Seeded source-shaped Gaussian noise is scaled to 30 dB over the full data volume. Each stage uses a low-pass waveform loss normalized by the filtered observed energy. The final filter tapers from 21 to 30 Hz. An onshore field study also used a 30 Hz finishing band in a different acquisition and inversion workflow. [Reta-Tang et al., 2023](https://doi.org/10.1190/tle42030190.1)

Two inputs use information available in this synthetic experiment. The upper velocity bound is the larger of 5,500 m/s and the maximum of the rescaled truth. The survey also stores its maximum surface speed. Gaussian growth uses the exact added noise, obtained by subtracting the simulated clean traces, to estimate noise energy separately at 6 and 9 Hz. The 0.30 s pre-arrival fallback has no usable window on the default survey. The filtered effective-sample count and per-atom charge are heuristics.

## Fitting and selection

Each low-frequency band has 325 gradient evaluations. Every tenth iteration, starting at zero, proposes shapes from connected, same-sign gradient regions. Cells at or below 1% of their region’s peak are cut before computing its weighted centre and covariance. The lagged local width floor is a quarter wavelength or one grid cell, whichever is larger. Centres may move outside the grid. The Gaussian parameters describe basis functions for the model.

For each proposed shape, the initial amplitude is 5% of local speed with the downhill sign. The fitter measures the bounded waveform loss for the probe and, when distinct, the capped quadratic estimate. It chooses the lower measured loss and keeps the probe on a tie. The gain must cover the component charge and exceed roundoff. Gaussian stage selection retains the best measured loss plus the per-atom charge among the explored states. Each grid band takes 350 Adam updates and returns its last bounded update. The loss history is measured before each update. The endpoint loss is measured on the saved grid. Related FWI work has also used adaptive basis parameterizations. [Adaptive meshless FWI](https://www.sciencedirect.com/science/article/pii/S0955799717300073)

## Paired comparisons

The fixed-Gaussian control adjusts amplitudes on prescribed isotropic lattices at 6 and 9 Hz and uses the same pixel refinement thereafter. Plain and smoothed-gradient pixel controls fit cells in all five bands. At 9 Hz count six trainable values per retained adaptive atom, one per fixed atom, or one per grid cell. Report model error, held-out waveform error and computational work separately. For a full rerun of the reported comparisons:

```sh
for model in marmousi overthrust; do
  for seed in 2 3; do
    for program in main.py comparisons/fixed_gaussians.py comparisons/pixels_plain.py comparisons/pixels_smoothed.py; do
      uv run --frozen python "$program" --model "$model" --noise-seed "$seed"
    done
  done
done
```

[Study results and raw numerical archives](results/README.md) accompany this version. To score the archived comparisons:

```sh
uv run --frozen python -m scripts.score_comparison results/runs/*/*.npz --output comparison.csv
```

Each run writes a timestamped directory in `outputs/` with `result.npz` and an unsmoothed velocity preview. The archive records input and observation hashes, software versions, source hashes, selection rules and wave counts. Band timers include setup and fitting, including any endpoint evaluation used for model selection. Separate forward calls used only to report a score are excluded. Timings from concurrent jobs are unsuitable for a wall-clock speed comparison. Use [scripts/make_figures.py](scripts/make_figures.py) to redraw an adaptive preview.

The optional Sigsbee input can be recreated locally from the distributor’s SEG-Y. See the [checksum-verified import instructions](data/README.md#optional-sigsbee-input).
