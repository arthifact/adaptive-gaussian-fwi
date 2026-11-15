"""Adaptive Gaussian FWI.

Modules, in the order main.py uses them:

  data_prepare.py       load and resample a velocity model
  simulation.py         survey, wave simulation, band-limited loss and gradient
  method_settings.py    fixed settings for the Gaussian stage
  gaussians.py          Gaussian velocity field
  moment_candidates.py  candidate Gaussians from gradient patches
  gaussians_fit.py      one band: growth, Adam updates, pruning, selection
  pixels.py             grid refinement at 13, 20 and 30 Hz

Baselines are in comparisons/, scoring, tables and figures in paper/.
"""
