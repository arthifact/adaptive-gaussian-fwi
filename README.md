# Adaptive Gaussian FWI

Code and results for *Adaptive Gaussian Parameterization for Multiscale Full-Waveform Inversion*.

A 2-D acoustic FWI where the 6 and 9 Hz macro model is a set of Gaussians added and shaped by rule, followed by pixel refinement at 13, 20 and 30 Hz.

## Run

```sh
uv sync --frozen
uv run --frozen python main.py --model marmousi --noise-seed 2
```

## Test

```sh
uv run --frozen python -m unittest discover -s tests
```

## Reproduce the paper

```sh
bash paper/run_experiments.sh tune
bash paper/run_experiments.sh main
bash paper/run_experiments.sh steps
bash paper/run_experiments.sh ablations
bash paper/run_experiments.sh tables
```

Tables are written to `paper/tables/`. Velocity model sources and licenses are in `data/NOTICE`.

## What the inversion knows

The true model is used only to make the data and to score the results. The inversion gets:

- the noisy shot gathers,
- the source wavelet (a 15 Hz Ricker),
- a separate noise-only record, from which it takes the noise level per band,
- a starting model that increases linearly from 1,500 to 3,000 m/s,
- velocity bounds of 1,000 to 7,000 m/s, the same for every model.

## How the data are made

- The observed data come from a grid of half the spacing (5 m) with 8th-order differences. The inversion predicts on the 10 m grid with 4th-order differences, so even the true model does not reproduce the data. Its misfit is given per case as `truth_heldout_loss` in `paper/tables/comparison.csv`.
- The noise is white and Gaussian, 20 dB below the RMS of the clean data. The source has little energy at low frequency, so the low bands are the noisiest: about 22 dB in the 6 Hz band, 27 dB at 9 Hz and 33 dB at 30 Hz.
- The boundaries absorb on every side, so there is no free surface.
- Each benchmark section is area-resampled as a whole onto 70 x 70 cells of 10 m. This keeps its shapes but not its real size or aspect ratio.

## How settings were chosen

All choices were made on Marmousi with noise seed 1, which is never reported. The reported cases are noise seeds 2, 3 and 4 of the four models.

| Choice | Tried on Marmousi seed 1 | Kept |
|---|---|---|
| TV weight (30 Hz RMSE, m/s) | 0.0001: 907, 0.001: 776, 0.01: 888, 0.1: 981 | 0.001 |
| Adam step of the fixed lattice (9 Hz RMSE, m/s) | 0.3: 590, 1: 534, 3: 554, 10: 630, 30: 671 | 1 m/s |
| Pruning of weak Gaussians (9 Hz RMSE, m/s) | on: 655, off: 595 | off |
| Growth threshold (9 Hz RMSE, m/s) | BIC: 595, times 0.5: 508, times 0.24: 565 | BIC |

The three thresholds do not order consistently on one case, so the BIC stays as the choice made before looking. The other settings of the adaptive method (candidates per round, trial amplitudes, Adam steps) were carried over unchanged from the earlier version of the code.

Two changes to the adaptive method were tried and rejected. Their runs are kept so the numbers can be read, but the code that made them is not in the repository, so they cannot be rerun:

| Change | Marmousi seed 1, 9 Hz RMSE (m/s) | Outcome |
|---|---|---|
| One Gaussian per growth round | 689 against 595 | rejected on the tuning seed (`results/marmousi-seed1/pilots/`) |
| One step scale per column instead of Adam's per-parameter scale | 606 against 595, widest Gaussian 295 m against 7.4 km | first adopted, then rejected (see below) |

The second change stops the drift described under known limits and looked harmless on the tuning seed. On the reported cases it was worse: Marmousi seeds 2 to 4 gave 607, 645 and 622 m/s against 629, 565 and 533, and Overthrust seed 2 gave 2,008 m/s, with the deep unit not built. Adam's steps were therefore kept. This is the one choice about the adaptive method that used reported cases; those runs are in `results/<case>/adaptive_column_steps/`.

Seeds 2 to 4 have now been looked at in detail, so any later change to the method should be tested on new seeds.

Overthrust, BP 2004 and SEAM took no part in any other choice; one Overthrust run with seed 1 was made only to see the effect of the wider velocity bound.

Because the lattice step is tuned on Marmousi only, `paper/run_experiments.sh steps` also runs the lattice with larger steps on every case, and `paper/tables/lattice-steps.csv` shows what each step gives. It also gives the lattice the adaptive method's shot budget at step 10 (`paper/tables/matched-work.csv`). Step 10 was picked after seeing that it is the lattice's best step on Overthrust, so this choice favours the lattice.

## Results

Velocity RMSE in m/s, mean of noise seeds 2, 3 and 4. Every number comes from `paper/tables/`.

Final models, after 30 Hz (`comparison.csv`):

| Model | Start | Adaptive Gaussians | Fixed lattice | Smoothed pixels | TV pixels | Plain pixels |
|---|---|---|---|---|---|---|
| Marmousi | 862 | 581 | 537 | 606 | 757 | 840 |
| Overthrust | 2,369 | 480 | 1,795 | 2,128 | 2,434 | 2,255 |
| BP 2004 | 1,530 | 436 | 462 | 1,120 | 1,346 | 1,501 |
| SEAM | 1,559 | 919 | 743 | 1,609 | 1,620 | 1,647 |
| All 12 cases | 1,580 | 604 | 884 | 1,366 | 1,539 | 1,561 |

Macro models after 9 Hz, with the lattice at other Adam steps (`lattice-steps.csv`) and at the adaptive method's shot count (`matched-work.csv`):

| Model | Adaptive | Lattice, step 1 | Step 3 | Step 10 | Step 30 | Step 1, equal work | Step 10, equal work |
|---|---|---|---|---|---|---|---|
| Marmousi | 576 | 546 | 623 | 728 | 817 | 607 | 790 |
| Overthrust | 445 | 1,606 | 1,429 | 783 | 766 | 1,704 | 726 |
| BP 2004 | 545 | 608 | 618 | 609 | 758 | 556 | 614 |
| SEAM | 978 | 817 | 454 | 1,028 | 1,197 | 548 | 1,100 |
| All 12 cases | 636 | 894 | 781 | 787 | 885 | 854 | 808 |
| Cases where the adaptive method is better | | 7 | 7 | 10 | 11 | 7 | 10 |

The same macro models by depth: top, middle and bottom third of the rows (`depth.csv`):

| Model | Adaptive | Fixed lattice, step 1 |
|---|---|---|
| Marmousi | 151, 448, 885 | 148, 472, 813 |
| Overthrust | 379, 441, 468 | 861, 1,680, 2,044 |
| BP 2004 | 560, 491, 538 | 553, 536, 720 |
| SEAM | 440, 592, 1,523 | 460, 568, 1,220 |

Misfit on four held-out shots, predicted on the fine grid, in units of 0.001 of the data energy (`fine_heldout_loss` in `comparison.csv`):

| Model | Band | Adaptive | Fixed lattice | Smoothed pixels | TV pixels | Plain pixels |
|---|---|---|---|---|---|---|
| Marmousi | 9 Hz | 1.6 | 2.0 | 0.6 | 2.9 | 3.1 |
| Overthrust | 9 Hz | 0.4 | 11.9 | 3.7 | 5.3 | 5.5 |
| BP 2004 | 9 Hz | 2.7 | 2.0 | 0.7 | 3.1 | 3.3 |
| SEAM | 9 Hz | 1.5 | 2.7 | 0.5 | 3.0 | 3.3 |
| Marmousi | 30 Hz | 8.2 | 8.1 | 2.6 | 6.9 | 8.6 |
| Overthrust | 30 Hz | 10.8 | 9.6 | 1.7 | 7.7 | 8.0 |
| BP 2004 | 30 Hz | 4.5 | 4.6 | 1.4 | 4.3 | 4.7 |
| SEAM | 30 Hz | 5.3 | 5.3 | 1.9 | 4.7 | 5.3 |

Ablations of the adaptive method after 9 Hz, with the mean number of Gaussians in brackets (`ablations.csv`):

| Model | Standard | Pruning on | Threshold halved | Threshold doubled |
|---|---|---|---|---|
| Marmousi | 576 (27) | 528 (24) | 562 (42) | 685 (16) |
| Overthrust | 445 (27) | 451 (26) | 528 (40) | 446 (20) |
| BP 2004 | 545 (57) | 672 (41) | 576 (71) | 517 (37) |
| SEAM | 978 (46) | 843 (43) | 573 (58) | 706 (30) |
| All 12 cases | 636 (39) | 624 (34) | 560 (53) | 588 (26) |
| Cases better than standard | | 4 | 6 | 7 |

What these show:

- Both Gaussian methods end much closer to the true velocity than the pixel-only methods on Overthrust, BP 2004 and SEAM. On Marmousi the starting ramp is already close and smoothed pixels are about as good.
- With one setting for every model, the adaptive method has the lowest mean error (636 m/s after 9 Hz). The lattice's best single step gives 781 m/s, and its best step changes with the model: 1 on Marmousi, 3 on SEAM, 10 to 30 on Overthrust.
- On Overthrust the deep fast unit (6,000 m/s) is now built without help from the bound. The adaptive macro model peaks at 6,210 and 6,628 m/s in seeds 2 and 4 and stays off the 7,000 m/s bound. In seed 3 it overshoots, and 17% of its cells sit on the bound.
- The Overthrust gap is not extra computation. With the adaptive method's shot budget the lattice reaches 726 m/s at step 10 and 1,704 m/s at step 1, against 445 m/s.
- With the best step for each case, which needs the true model to choose, the lattice averages 567 m/s and is better than the adaptive method in 6 of the 12 cases.
- The adaptive method repeats less well across noise seeds. On SEAM it gives 627 to 1,364 m/s against 779 to 862 for the lattice. The difference is in the bottom third, which the five shots barely light. In SEAM seed 3 a single Gaussian of -2,228 m/s sits under the salt and the data do not reject it.
- No ablation is better in a consistent way, so the settings stay as chosen on the tuning seed. But single cases move a lot: halving the threshold takes SEAM seed 3 from 1,364 to 414 m/s, and doubling it takes Marmousi seed 4 from 533 to 933. The error of one adaptive run is therefore uncertain by a few hundred m/s, and only the means over cases should be compared.
- Velocity error and data fit do not agree. Smoothed pixels predict the held-out shots best at 30 Hz on every model, with a velocity error up to four times larger. A likely reason, not tested here, is that the plain pixel stage after both Gaussian methods fits cell-scale detail that does not carry over to the fine grid.

## Known limits

- Synthetic, 2-D, acoustic, constant density, with a known source wavelet.
- Five shots and a 690 m aperture, so the deep part of each model is poorly illuminated.
- Adam moves every parameter at about its rate whatever the size of its gradient. This is what builds deep structure in both Gaussian methods, and it makes their deep velocities depend on the step size and the iteration count more than on the data. It also lets a Gaussian that the data barely constrain drift: some end several kilometres wide or centred outside the grid.
- The 10 m grid cannot follow cell-scale velocity jumps exactly, so the true model itself misfits the fine-grid data. This modeling error is about half the noise energy in the 6 Hz band, about twice the noise at 9 Hz, and 15 to 20 times the noise at 30 Hz. The growth threshold only accounts for the noise.
