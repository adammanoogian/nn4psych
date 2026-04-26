#!/bin/bash
# =============================================================================
# Submit Two Parallel Per-Context Latent Circuit Fits + Autopush
# =============================================================================
# Plan 03-06 (Gap 3 diagnostic): fits a LatentNet ensemble independently for
# modality_context=0 and modality_context=1, then runs autopush once both
# finish (afterany so push fires even if one fit fails).
#
# Usage:
#   bash cluster/run_per_context_fits.sh
#
#   # Override parameters:
#   N_INITS=50 EPOCHS=300 bash cluster/run_per_context_fits.sh
#
#   # Email notification when push completes:
#   NOTIFY_EMAIL=adam.manoogian@monash.edu bash cluster/run_per_context_fits.sh
#
# Output:
#   output/circuit_analysis/per_context/context_0/   (fit for context 0)
#   output/circuit_analysis/per_context/context_1/   (fit for context 1)
#   cluster/logs/per_context_jobs.txt                (JID records)
# =============================================================================

set -euo pipefail

# =============================================================================
# CRLF strip — required when repo was cloned on Windows
# =============================================================================
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh

# =============================================================================
# Parameters (override via env vars)
# =============================================================================
N_LATENT=${N_LATENT:-12}
N_INITS=${N_INITS:-100}
EPOCHS=${EPOCHS:-500}
NOTIFY_EMAIL=${NOTIFY_EMAIL:-}

echo "============================================================"
echo "Per-Context Latent Circuit Fit Submitter"
echo "============================================================"
echo "n_latent: $N_LATENT   n_inits: $N_INITS   epochs: $EPOCHS"
echo "Output:   output/circuit_analysis/per_context/context_{0,1}/"
echo "============================================================"
echo ""

mkdir -p cluster/logs

# =============================================================================
# Submit context 0 fit
# =============================================================================
JID0=$(sbatch --parsable \
    --job-name=circuit_per_ctx0 \
    --export=ALL,CONTEXT_ID=0,N_LATENT="${N_LATENT}",N_INITS="${N_INITS}",EPOCHS="${EPOCHS}" \
    cluster/run_per_context_one.slurm)
echo "Submitted context 0 fit: JID=$JID0"

# =============================================================================
# Submit context 1 fit
# =============================================================================
JID1=$(sbatch --parsable \
    --job-name=circuit_per_ctx1 \
    --export=ALL,CONTEXT_ID=1,N_LATENT="${N_LATENT}",N_INITS="${N_INITS}",EPOCHS="${EPOCHS}" \
    cluster/run_per_context_one.slurm)
echo "Submitted context 1 fit: JID=$JID1"

# =============================================================================
# Submit autopush dependent on both fits finishing
# =============================================================================
PUSH_EXPORT="ALL,PARENT_JOBS=${JID0}:${JID1}"
if [[ -n "$NOTIFY_EMAIL" ]]; then
    PUSH_EXPORT="${PUSH_EXPORT},NOTIFY_EMAIL=${NOTIFY_EMAIL}"
fi

PUSH_JID=$(sbatch --parsable \
    --dependency=afterany:"${JID0}":"${JID1}" \
    --export="${PUSH_EXPORT}" \
    cluster/99_push_results.slurm)
echo "Submitted autopush:      JID=$PUSH_JID (dependency=afterany:${JID0}:${JID1})"

# =============================================================================
# Record JIDs for SUMMARY and continuation agent
# =============================================================================
LOG_FILE="cluster/logs/per_context_jobs.txt"
{
    echo "# Per-context fit jobs — Plan 03-06"
    echo "# Submitted: $(date --iso-8601=seconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)"
    echo "context_0_fit_jid=${JID0}"
    echo "context_1_fit_jid=${JID1}"
    echo "autopush_jid=${PUSH_JID}"
} > "$LOG_FILE"
echo ""
echo "JIDs recorded → $LOG_FILE"
echo ""

echo "============================================================"
echo "All jobs submitted."
echo "  Context 0 fit:  $JID0  (~95 min)"
echo "  Context 1 fit:  $JID1  (~95 min, parallel)"
echo "  Autopush:       $PUSH_JID  (fires after both finish)"
echo ""
echo "Monitor:  squeue -j ${JID0},${JID1},${PUSH_JID}"
echo "Logs:     cluster/logs/circuit_per_ctx0_${JID0}.out"
echo "          cluster/logs/circuit_per_ctx1_${JID1}.out"
echo "============================================================"
