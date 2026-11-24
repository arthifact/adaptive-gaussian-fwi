#!/usr/bin/env bash
# All runs for the paper. From the project root, in order:
#
#   bash paper/run_experiments.sh tune        # 1. choose the TV weight, the lattice step, pruning and the threshold (about 20 min)
#   bash paper/run_experiments.sh main        # 2. every method on every model (about 3 h)
#   bash paper/run_experiments.sh steps       # 3. fixed lattice with larger steps (about 1 h)
#   bash paper/run_experiments.sh ablations   # 4. pruning and threshold checks (about 1 h)
#   bash paper/run_experiments.sh tables      # 5. rebuild every table (about 3 min)
#
# Times on a 14-core laptop. Jobs in a batch run in parallel; logs in paper/logs/.
# 'main' overwrites earlier runs (older versions are in git history).
set -euo pipefail
cd "$(dirname "$0")/.."
LOGS=paper/logs
mkdir -p "$LOGS"

MODELS="marmousi overthrust bp_salt seam"
SEEDS="2 3 4"
TUNING_FRACTIONS="0.0001 0.001 0.01 0.1"
TUNING_RATES="0.3 1 3 10 30"  # Adam step of the fixed-lattice amplitudes, m/s
TUNING_THRESHOLDS="0.5 0.24"  # scale of the growth threshold; 1 is the BIC
OTHER_RATES="3 10 30"  # larger steps, run on every case to show the lattice at its best
MATCHED_RATE="10"  # larger step that also gets the adaptive shot budget

# Run each argument in parallel, one log per command; fail if any fails.
run_parallel() {
  local pids="" command log failed=0
  for command in "$@"; do
    log="$LOGS/$(echo "$command" | tr -cs 'A-Za-z0-9._-' '_').log"
    echo "  start: $command"
    bash -c "$command" > "$log" 2>&1 &
    pids="$pids $!"
  done
  for pid in $pids; do wait "$pid" || failed=1; done
  if [ "$failed" -ne 0 ]; then echo "A run failed; see $LOGS/"; exit 1; fi
}

PY="${PY:-uv run --frozen python}"  # set PY to use another interpreter

case "${1:-}" in
  tune)
    # Marmousi seed 1 is only used for tuning and is never reported.
    echo "Tuning the TV weight, the fixed-lattice step and the adaptive method on Marmousi, noise seed 1"
    commands=()
    for fraction in $TUNING_FRACTIONS; do
      commands+=("$PY comparisons/pixels_tv.py --model marmousi --noise-seed 1 --tv-fraction $fraction")
    done
    for rate in $TUNING_RATES; do
      commands+=("$PY comparisons/fixed_gaussians.py --model marmousi --noise-seed 1 --macro-only --rate $rate")
    done
    commands+=("$PY main.py --model marmousi --noise-seed 1 --macro-only")
    commands+=("$PY main.py --model marmousi --noise-seed 1 --macro-only --pruning")
    for scale in $TUNING_THRESHOLDS; do
      commands+=("$PY main.py --model marmousi --noise-seed 1 --macro-only --threshold-scale $scale")
    done
    run_parallel "${commands[@]}"
    $PY - <<'EOF'
import json, numpy as np
from pathlib import Path
from comparisons.pixels_tv import TV_FRACTION
results = []
for folder in Path("comparisons/results/marmousi-seed1").glob("pixels_tv*"):
    with np.load(folder / "result.npz") as data:
        settings = json.loads(data["settings"].item())
        rmse = json.loads(data["progress"].item())[-1]["rmse"]
    results.append((settings["regularization"]["fraction_of_rms_data_gradient"], rmse))
print("30 Hz velocity RMSE on Marmousi seed 1 (lower is better):")
for fraction, rmse in sorted(results):
    print(f"  fraction {fraction:<5g} {rmse:8.1f} m/s")
print(f"Best: {min(results, key=lambda item: item[1])[0]:g}")
print(f"Set TV_FRACTION in comparisons/pixels_tv.py to the best fraction (now {TV_FRACTION:g}).")
from comparisons.fixed_gaussians import MACRO_RATE
results = []
for folder in Path("comparisons/results/marmousi-seed1").glob("fixed_gaussians*_macro"):
    with np.load(folder / "result.npz") as data:
        settings = json.loads(data["settings"].item())
        rmse = json.loads(data["progress"].item())[-1]["rmse"]
    results.append((settings["macro_adam_rate"], rmse))
print("9 Hz velocity RMSE of the fixed lattice on Marmousi seed 1 (lower is better):")
for rate, rmse in sorted(results):
    print(f"  step {rate:<5g} {rmse:8.1f} m/s")
print(f"Best: {min(results, key=lambda item: item[1])[0]:g}")
print(f"Set MACRO_RATE in comparisons/fixed_gaussians.py to the best step (now {MACRO_RATE:g}), then run 'main'.")
print("9 Hz velocity RMSE of the adaptive method on Marmousi seed 1 (lower is better):")
for folder in sorted(Path("results/marmousi-seed1").glob("adaptive*_macro")):
    with np.load(folder / "result.npz") as data:
        rmse = json.loads(data["progress"].item())[-1]["rmse"]
    print(f"  {folder.name:<30} {rmse:8.1f} m/s")
EOF
    ;;
  main)
    echo "TV fraction in use: $($PY -c 'from comparisons.pixels_tv import TV_FRACTION; print(TV_FRACTION)')"
    for model in $MODELS; do
      for seed in $SEEDS; do
        echo "All methods: $model, seed $seed"
        run_parallel \
          "$PY main.py --model $model --noise-seed $seed" \
          "$PY comparisons/fixed_gaussians.py --model $model --noise-seed $seed" \
          "$PY comparisons/pixels_plain.py --model $model --noise-seed $seed" \
          "$PY comparisons/pixels_smoothed.py --model $model --noise-seed $seed" \
          "$PY comparisons/pixels_tv.py --model $model --noise-seed $seed"
      done
    done
    # Needs the adaptive runs above for their shot counts.
    echo "Work-matched fixed lattice for every case"
    commands=()
    for model in $MODELS; do
      for seed in $SEEDS; do
        commands+=("$PY -m paper.match_macro_work results/$model-seed$seed/adaptive")
      done
    done
    run_parallel "${commands[@]}"
    ;;
  steps)
    # The reported lattice uses the step tuned on Marmousi seed 1. These runs show
    # what a different step would give on each case (6 and 9 Hz only).
    for model in $MODELS; do
      echo "Fixed lattice steps: $model"
      commands=()
      for seed in $SEEDS; do
        for rate in $OTHER_RATES; do
          commands+=("$PY comparisons/fixed_gaussians.py --model $model --noise-seed $seed --macro-only --rate $rate")
        done
      done
      run_parallel "${commands[@]}"
    done
    # Step 1 is too small for Overthrust, so equal work is also given at a larger step.
    echo "Work-matched fixed lattice at step $MATCHED_RATE"
    commands=()
    for model in $MODELS; do
      for seed in $SEEDS; do
        commands+=("$PY -m paper.match_macro_work results/$model-seed$seed/adaptive --rate $MATCHED_RATE")
      done
    done
    run_parallel "${commands[@]}"
    ;;
  ablations)
    # Macro-only (6 and 9 Hz); these settings don't affect the pixel stage.
    for model in $MODELS; do
      echo "Ablations: $model"
      commands=()
      for seed in $SEEDS; do
        commands+=("$PY main.py --model $model --noise-seed $seed --macro-only --pruning")
        commands+=("$PY main.py --model $model --noise-seed $seed --macro-only --threshold-scale 0.5")
        commands+=("$PY main.py --model $model --noise-seed $seed --macro-only --threshold-scale 2")
      done
      run_parallel "${commands[@]}"
    done
    ;;
  tables)
    $PY -m paper.score_comparison
    $PY -m paper.tabulate_matched_work
    $PY -m paper.tabulate_ablations
    $PY -m paper.tabulate_lattice_steps
    $PY -m paper.tabulate_depth
    $PY -m paper.make_figures
    echo "Tables are in paper/tables/."
    ;;
  *)
    echo "Usage: bash paper/run_experiments.sh {tune|main|steps|ablations|tables}"
    exit 2
    ;;
esac
