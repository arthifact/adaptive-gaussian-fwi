# Adaptive Gaussian FWI

Code and results for *Adaptive Gaussian Parameterization for Multiscale Full-Waveform Inversion*.

A 2-D acoustic FWI where the 6 and 9 Hz macro model is a set of Gaussians added, shaped and removed by rule, followed by pixel refinement at 13, 20 and 30 Hz.

## Run

```sh
uv sync --frozen
uv run --frozen python main.py --model marmousi --noise-seed 2
```

## Reproduce the paper

```sh
bash paper/run_experiments.sh tune
bash paper/run_experiments.sh main
bash paper/run_experiments.sh ablations
bash paper/run_experiments.sh tables
```

Tables are written to `paper/tables/`. Velocity model sources and licenses are in `data/NOTICE`.
