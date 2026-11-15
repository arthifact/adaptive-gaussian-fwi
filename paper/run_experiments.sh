#!/usr/bin/env bash
# All runs for the paper. From the project root, in order:
#
#   bash paper/run_experiments.sh tune        # 1. choose the TV weight (about 15 min)
#   bash paper/run_experiments.sh main        # 2. every method on every model (about 3 h)
#   bash paper/run_experiments.sh ablations   # 3. pruning and threshold checks (about 1 h)
#   bash paper/run_experiments.sh tables      # 4. rebuild every table (about 3 min)
#
# Times on a 14-core laptop. Jobs in a batch run in parallel; logs in paper/logs/.
# 'main' overwrites earlier runs (older versions are in git history).
set -euo pipefail
cd "$(dirname "$0")/.."
LOGS=paper/logs
mkdir -p "$LOGS"

MODELS="marmousi overthrust bp_salt seam"
SEEDS="2 3 4"
TUNING_FRACTIONS="0.001 0.003 0.01 0.03 0.1 0.3 1"

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

PY="uv run --frozen python"

case "${1:-}" in
  tune)
    # Marmousi seed 1 is only used for tuning and is never reported.
    echo "Tuning the TV weight on Marmousi, noise seed 1"
    commands=()
    for fraction in $TUNING_FRACTIONS; do
      commands+=("$PY comparisons/pixels_tv.py --model marmousi --noise-seed 1 --tv-fraction $fraction")
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
print(f"Set TV_FRACTION in comparisons/pixels_tv.py to the best fraction (now {TV_FRACTION:g}), then run 'main'.")
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
  ablations)
    # Macro-only (6 and 9 Hz); these settings don't affect the pixel stage.
    for model in $MODELS; do
      echo "Ablations: $model"
      commands=()
      for seed in $SEEDS; do
        commands+=("$PY main.py --model $model --noise-seed $seed --macro-only --no-pruning")
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
    $PY -m paper.make_figures
    echo "Tables are in paper/tables/."
    ;;
  *)
    echo "Usage: bash paper/run_experiments.sh {tune|main|ablations|tables}"
    exit 2
    ;;
esac
