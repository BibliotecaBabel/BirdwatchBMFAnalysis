#!/bin/bash
# Run the seven sweeps behind the paper's figures, one after the other.
# Each sweep is parallelized over its configurations with Hydra's joblib
# launcher; set N_JOBS to change the number of parallel jobs (default 8).
# Finished experiments are skipped, so an interrupted run can be restarted.
#
# Four configs sweep more values than the figures show. The override after
# the config name keeps the values the figures read (131 experiments in
# total); delete it to run that config's whole grid (266 in total).

N_JOBS="${N_JOBS:-8}"

sweeps=(
    "agent-mean-sweep"
    "agent-mean-sweep-binary model.reg_lambda=0.01"
    "item-mean-sweep"
    "kappa-sweep"
    "robustness-distortion-500 model.reg_intercept_multiplier=1.0"
    "misspecification-sweep model.reg_lambda=0.01"
    "confounder-sweep dg.confounder_item_std=1.0"
)

failed=0
for sweep in "${sweeps[@]}"; do
    read -r config override <<< "$sweep"
    echo
    echo "=== $config $override (started $(date '+%Y-%m-%d %H:%M:%S')) ==="
    if ! poetry run python run.py -m --config-name "$config" $override hydra.launcher.n_jobs="$N_JOBS"; then
        echo "sweep '$config' failed; continuing with the next one"
        failed=$((failed + 1))
    fi
done

echo
if [ "$failed" -gt 0 ]; then
    echo "$failed sweep(s) failed"
    exit 1
fi
echo "all sweeps completed"
