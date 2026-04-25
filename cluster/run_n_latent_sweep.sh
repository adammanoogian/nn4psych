#!/bin/bash
# =============================================================================
# Wave A: n_latent Sweep Submitter
# =============================================================================
# Submits 4 cluster/run_circuit_ensemble.sh jobs, one per n_latent in
# {4, 8, 12, 16}. Each job writes to output/circuit_analysis/n_latent_sweep/n{N}/.
#
# n_latent=4 will fail LatentNet's connectivity_masks() assertion
# (requires n >= max(input_size=7, output_size=3) = 7). The sweep submits it
# anyway so the failure is recorded in the cluster log; the aggregator
# (Task 3) will note any missing per-rank artifacts.
#
# Usage:
#   bash cluster/run_n_latent_sweep.sh
#   bash cluster/run_n_latent_sweep.sh --ranks 8,12,16   # skip n=4
#
# Job IDs are printed to stdout and saved to cluster/logs/n_latent_sweep_jobs.txt
# for later squeue / sacct inspection.
# =============================================================================

set -euo pipefail

RANKS="${1:-4,8,12,16}"
if [[ "$RANKS" == --ranks ]]; then
    RANKS="${2:-4,8,12,16}"
fi
# Allow --ranks=4,8,12,16 form too
RANKS="${RANKS#--ranks=}"

mkdir -p cluster/logs
JOBS_FILE="cluster/logs/n_latent_sweep_jobs.txt"
: > "$JOBS_FILE"

echo "Submitting n_latent sweep over ranks: $RANKS"
echo "Each job writes to output/circuit_analysis/n_latent_sweep/n{N}/"
echo ""

IFS=',' read -ra RANK_ARRAY <<< "$RANKS"
for N in "${RANK_ARRAY[@]}"; do
    SUBDIR="n_latent_sweep/n${N}"
    JOB_NAME="circuit_n${N}"

    # FORCE_RECOLLECT=0: all ranks reuse the same circuit_data.npz
    # (the .regen.lock in run_circuit_ensemble.sh handles concurrent reads safely)
    JOB_ID=$(sbatch \
        --job-name="$JOB_NAME" \
        --output="cluster/logs/${JOB_NAME}_%j.out" \
        --error="cluster/logs/${JOB_NAME}_%j.err" \
        --export=ALL,N_LATENT="$N",OUTPUT_SUBDIR="$SUBDIR",FORCE_RECOLLECT=0 \
        cluster/run_circuit_ensemble.sh \
        | awk '{print $NF}')

    echo "  n_latent=$N -> job $JOB_ID, output -> output/circuit_analysis/$SUBDIR/"
    echo "$N $JOB_ID $SUBDIR" >> "$JOBS_FILE"
done

echo ""
echo "All jobs submitted. Job IDs saved to $JOBS_FILE"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f cluster/logs/circuit_n*_*.out"
echo ""
echo "When all jobs complete, run aggregator:"
echo "  python scripts/analysis/aggregate_n_latent_sweep.py"
