#!/bin/bash
# =============================================================================
# Wave A: n_latent Sweep Submitter (with auto-push)
# =============================================================================
# Submits 4 cluster/run_circuit_ensemble.sh jobs, one per n_latent in
# {4, 8, 12, 16}. Each job writes to output/circuit_analysis/n_latent_sweep/n{N}/.
# After all 4 jobs complete (afterany), submits cluster/99_push_results.slurm
# to commit + push the per-rank artifacts to GIT_REMOTE on a results branch.
#
# n_latent=4 will fail LatentNet's connectivity_masks() assertion
# (requires n >= max(input_size=7, output_size=3) = 7). The sweep submits it
# anyway so the failure is recorded in the cluster log; the aggregator
# (scripts/analysis/aggregate_n_latent_sweep.py) notes any missing per-rank
# artifacts. afterany (NOT afterok) ensures the push job still fires even if
# n=4 hits the expected assertion.
#
# Usage:
#   bash cluster/run_n_latent_sweep.sh                   # ranks 4,8,12,16 + autopush
#   bash cluster/run_n_latent_sweep.sh --ranks 8,12,16   # skip n=4
#   bash cluster/run_n_latent_sweep.sh --no-autopush     # skip the push job
#
#   # Push directly to main on completion (single-author shortcut, riskier):
#   PUSH_TO_MAIN=true bash cluster/run_n_latent_sweep.sh
#
#   # Custom remote (if origin is not your fork):
#   GIT_REMOTE=upstream bash cluster/run_n_latent_sweep.sh
#
#   # Custom email notification on push completion:
#   NOTIFY_EMAIL=adam.manoogian@monash.edu bash cluster/run_n_latent_sweep.sh
#
# Job IDs are printed to stdout and saved to cluster/logs/n_latent_sweep_jobs.txt
# for later squeue / sacct inspection. The push job ID is appended.
# =============================================================================

set -euo pipefail

# Strip Windows CRLF from cluster scripts before any sbatch call (idempotent).
# Required because the repo was cloned from Windows; sbatch rejects \r in #!.
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
RANKS="4,8,12,16"
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
            sed -n '2,/^# ===/p' "$0" | head -n 32
            exit 0
            ;;
        [0-9]*)
            # Positional: treat as ranks (e.g. `bash run_n_latent_sweep.sh 4,8,12,16`)
            RANKS="$1"
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Try: bash cluster/run_n_latent_sweep.sh --help" >&2
            exit 2
            ;;
    esac
done

mkdir -p cluster/logs
JOBS_FILE="cluster/logs/n_latent_sweep_jobs.txt"
: > "$JOBS_FILE"

echo "Submitting n_latent sweep over ranks: $RANKS"
echo "Each job writes to output/circuit_analysis/n_latent_sweep/n{N}/"
echo "Auto-push: $AUTOPUSH"
echo ""

# -----------------------------------------------------------------------------
# Submit per-rank fitting jobs (--parsable for clean dependency chaining)
# -----------------------------------------------------------------------------
FIT_JOB_IDS=()
IFS=',' read -ra RANK_ARRAY <<< "$RANKS"
for N in "${RANK_ARRAY[@]}"; do
    SUBDIR="n_latent_sweep/n${N}"
    JOB_NAME="circuit_n${N}"

    # FORCE_RECOLLECT=0: all ranks reuse the same circuit_data.npz
    # (the .regen.lock in run_circuit_ensemble.sh handles concurrent reads safely)
    JOB_ID=$(sbatch --parsable \
        --job-name="$JOB_NAME" \
        --output="cluster/logs/${JOB_NAME}_%j.out" \
        --error="cluster/logs/${JOB_NAME}_%j.err" \
        --export=ALL,N_LATENT="$N",OUTPUT_SUBDIR="$SUBDIR",FORCE_RECOLLECT=0 \
        cluster/run_circuit_ensemble.sh)

    echo "  n_latent=$N -> job $JOB_ID, output -> output/circuit_analysis/$SUBDIR/"
    echo "$N $JOB_ID $SUBDIR" >> "$JOBS_FILE"
    FIT_JOB_IDS+=("$JOB_ID")
done

echo ""
echo "All ${#FIT_JOB_IDS[@]} fitting jobs submitted. Job IDs saved to $JOBS_FILE"

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
echo "  tail -f cluster/logs/circuit_n*_*.out"
if [[ "$AUTOPUSH" == "true" ]]; then
    echo "  tail -f cluster/logs/push_results_*.out   # push job log"
fi
echo ""
echo "When all jobs complete, run aggregator (locally, after the push lands):"
echo "  git fetch origin && git merge origin/results/slurm-<timestamp>   # if branch strategy"
echo "  python scripts/analysis/aggregate_n_latent_sweep.py"
