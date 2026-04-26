#!/bin/bash
# =============================================================================
# Gap 1 (03-05): Masked-Loss n_latent Sweep Submitter (with auto-push)
# =============================================================================
# Submits 3 cluster/run_circuit_ensemble.sh jobs, one per n_latent in
# {8, 12, 16}, each with MASKED=1. Each job writes to
# output/circuit_analysis/n_latent_sweep_masked/n{N}/.
# After all 3 jobs complete (afterany), submits cluster/99_push_results.slurm
# to commit + push the per-rank masked artifacts to GIT_REMOTE on a results branch.
#
# n_latent=4 is omitted (failed LatentNet assertion in Wave A; n=4 < max(7,3)=7).
# afterany (NOT afterok) ensures the push job still fires even if a fit fails.
#
# IMPORTANT: run_circuit_ensemble.sh checks that circuit_data.npz contains
# 'task_active_mask' when MASKED=1. Ensure you have pushed the updated
# circuit_data.npz (with task_active_mask, from Plan 03-05 Task 1D) to
# origin/main and that the cluster has pulled it BEFORE submitting this sweep.
#
# Usage:
#   bash cluster/run_n_latent_sweep_masked.sh                 # ranks 8,12,16 + autopush
#   bash cluster/run_n_latent_sweep_masked.sh --ranks 12,16   # custom subset
#   bash cluster/run_n_latent_sweep_masked.sh --no-autopush   # skip the push job
#
#   # Push directly to main on completion (single-author shortcut, riskier):
#   PUSH_TO_MAIN=true bash cluster/run_n_latent_sweep_masked.sh
#
#   # Custom remote (if origin is not your fork):
#   GIT_REMOTE=upstream bash cluster/run_n_latent_sweep_masked.sh
#
#   # Custom email notification on push completion:
#   NOTIFY_EMAIL=adam.manoogian@monash.edu bash cluster/run_n_latent_sweep_masked.sh
#
# Job IDs are printed to stdout and saved to cluster/logs/n_latent_sweep_masked_jobs.txt
# for later squeue / sacct inspection. The push job ID is appended.
# =============================================================================

set -euo pipefail

# Strip Windows CRLF from cluster scripts before any sbatch call (idempotent).
# Required because the repo was cloned from Windows; sbatch rejects \r in #!.
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
RANKS="8,12,16"
AUTOPUSH=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ranks)
            RANKS="${2:?--ranks requires a value, e.g. --ranks 8,12,16}"
            shift 2
            ;;
        --ranks=*)
            RANKS="${1#--ranks=}"
            shift
            ;;
        --no-autopush)
            AUTOPUSH=false
            shift
            ;;
        -h|--help)
            sed -n '2,/^# ===/p' "$0" | head -n 38
            exit 0
            ;;
        [0-9]*)
            # Positional: treat as ranks (e.g. `bash run_n_latent_sweep_masked.sh 8,12,16`)
            RANKS="$1"
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Try: bash cluster/run_n_latent_sweep_masked.sh --help" >&2
            exit 2
            ;;
    esac
done

mkdir -p cluster/logs
JOBS_FILE="cluster/logs/n_latent_sweep_masked_jobs.txt"
: > "$JOBS_FILE"

echo "Submitting masked n_latent sweep over ranks: $RANKS"
echo "Each job writes to output/circuit_analysis/n_latent_sweep_masked/n{N}/"
echo "MASKED=1 exported to run_circuit_ensemble.sh (triggers masked-loss --masked flag)"
echo "Auto-push: $AUTOPUSH"
echo ""

# -----------------------------------------------------------------------------
# Submit per-rank fitting jobs (--parsable for clean dependency chaining)
# -----------------------------------------------------------------------------
FIT_JOB_IDS=()
IFS=',' read -ra RANK_ARRAY <<< "$RANKS"
for N in "${RANK_ARRAY[@]}"; do
    SUBDIR="n_latent_sweep_masked/n${N}"
    JOB_NAME="circuit_masked_n${N}"

    # MASKED=1: triggers --masked in the inner Python script (see run_circuit_ensemble.sh)
    # FORCE_RECOLLECT=0: all ranks reuse the same circuit_data.npz (pulled from repo)
    JOB_ID=$(sbatch --parsable \
        --job-name="$JOB_NAME" \
        --output="cluster/logs/${JOB_NAME}_%j.out" \
        --error="cluster/logs/${JOB_NAME}_%j.err" \
        --export=ALL,N_LATENT="$N",OUTPUT_SUBDIR="$SUBDIR",FORCE_RECOLLECT=0,MASKED=1 \
        cluster/run_circuit_ensemble.sh)

    echo "  n_latent=$N -> job $JOB_ID, output -> output/circuit_analysis/$SUBDIR/"
    echo "$N $JOB_ID $SUBDIR MASKED=1" >> "$JOBS_FILE"
    FIT_JOB_IDS+=("$JOB_ID")
done

echo ""
echo "All ${#FIT_JOB_IDS[@]} masked fitting jobs submitted. Job IDs saved to $JOBS_FILE"

# -----------------------------------------------------------------------------
# Submit dependent push job (afterany — fires even if some fits fail)
# -----------------------------------------------------------------------------
if [[ "$AUTOPUSH" == "true" && ${#FIT_JOB_IDS[@]} -gt 0 ]]; then
    FIT_DEP=$(IFS=:; echo "${FIT_JOB_IDS[*]}")
    PARENT_LIST=$(IFS=' '; echo "${FIT_JOB_IDS[*]}")

    # Pass through optional env vars set by the caller (PUSH_TO_MAIN, GIT_REMOTE,
    # NOTIFY_EMAIL, BRANCH_NAME). Empty values are harmless — the push script
    # has its own defaults.
    PUSH_EXPORT="ALL,PARENT_JOBS=${PARENT_LIST}"
    [[ -n "${PUSH_TO_MAIN:-}" ]] && PUSH_EXPORT="${PUSH_EXPORT},PUSH_TO_MAIN=${PUSH_TO_MAIN}"
    [[ -n "${GIT_REMOTE:-}" ]] && PUSH_EXPORT="${PUSH_EXPORT},GIT_REMOTE=${GIT_REMOTE}"
    [[ -n "${NOTIFY_EMAIL:-}" ]] && PUSH_EXPORT="${PUSH_EXPORT},NOTIFY_EMAIL=${NOTIFY_EMAIL}"
    [[ -n "${BRANCH_NAME:-}" ]] && PUSH_EXPORT="${PUSH_EXPORT},BRANCH_NAME=${BRANCH_NAME}"

    PUSH_JOB_ID=$(sbatch --parsable \
        --dependency=afterany:"${FIT_DEP}" \
        --export="${PUSH_EXPORT}" \
        cluster/99_push_results.slurm)

    echo "  push  -> job $PUSH_JOB_ID (depends on afterany:${FIT_DEP})"
    echo "PUSH $PUSH_JOB_ID afterany:${FIT_DEP}" >> "$JOBS_FILE"
else
    echo "  (auto-push disabled; pull artifacts manually after jobs finish)"
fi

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f cluster/logs/circuit_masked_n*_*.out"
if [[ "$AUTOPUSH" == "true" ]]; then
    echo "  tail -f cluster/logs/push_results_*.out   # push job log"
fi
echo ""
echo "After autopush lands, run aggregator locally:"
echo "  git pull origin main"
echo "  python scripts/analysis/aggregate_n_latent_sweep_masked.py"
