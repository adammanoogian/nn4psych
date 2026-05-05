#!/bin/bash
# =============================================================================
# Phase 4 Bayesian pipeline orchestrator (cluster)
# =============================================================================
# Submits the Phase 4 Bayesian validation jobs to SLURM with proper
# dependencies and autopush wiring:
#
#   (A) 09a_param_recovery_smoke.slurm     ← independent
#   (B) 09b_fit_human_subjects.slurm       ← independent (--array=0-267)
#   (C) 09c_aggregate_human_fits.slurm     ← afterok:B
#   (D) 99_push_results.slurm              ← afterany:A:C (push everything)
#
# Run from the cluster login node:
#   bash cluster/run_phase4_bayesian.sh
#
# Optional environment overrides:
#   SKIP_SMOKE=1       — skip job (A); only run subject fits + aggregation
#   SKIP_FITS=1        — skip jobs (B) and (C); only run smoke + push
#   ARRAY_RANGE="0-9"  — limit array (smoke testing the SLURM wiring)
#   PUSH_TO_MAIN=true  — push to main instead of a branch
#   GIT_REMOTE=upstream — override remote
#
# This script handles the Windows CRLF gotcha (fixes line endings on every
# .slurm/.sh in cluster/ before any sbatch call), per the comment in
# 99_push_results.slurm.
#
# Output: prints job IDs and the chained dependency graph.
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# =============================================================================
# CRLF safety (repo cloned on Windows)
# =============================================================================
echo "Fixing CRLF in cluster/*.slurm and cluster/*.sh ..."
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true

# =============================================================================
# Submission
# =============================================================================
mkdir -p cluster/logs

PARENT_JOBS=""

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
    SMOKE_JID=$(sbatch --parsable cluster/09a_param_recovery_smoke.slurm)
    echo "  (A) smoke recovery       JID=$SMOKE_JID"
    PARENT_JOBS="${PARENT_JOBS:+$PARENT_JOBS:}$SMOKE_JID"
else
    echo "  (A) smoke recovery       SKIPPED (SKIP_SMOKE=1)"
    SMOKE_JID=""
fi

if [[ "${SKIP_FITS:-0}" != "1" ]]; then
    ARRAY_RANGE="${ARRAY_RANGE:-0-267}"
    FIT_JID=$(sbatch --parsable --array="$ARRAY_RANGE" \
        cluster/09b_fit_human_subjects.slurm)
    echo "  (B) per-subject fits     JID=$FIT_JID  array=$ARRAY_RANGE"

    AGG_JID=$(sbatch --parsable --dependency=afterok:"$FIT_JID" \
        cluster/09c_aggregate_human_fits.slurm)
    echo "  (C) aggregate            JID=$AGG_JID  (afterok:$FIT_JID)"

    PARENT_JOBS="${PARENT_JOBS:+$PARENT_JOBS:}$AGG_JID"
else
    echo "  (B) per-subject fits     SKIPPED (SKIP_FITS=1)"
    echo "  (C) aggregate            SKIPPED (SKIP_FITS=1)"
fi

# =============================================================================
# Autopush — runs afterany so it pushes whatever is available, even if some
# upstream jobs failed (we can inspect logs to diagnose, vs. having results
# stranded on the cluster).
# =============================================================================
if [[ -n "$PARENT_JOBS" ]]; then
    PUSH_EXPORTS="ALL,PARENT_JOBS=$PARENT_JOBS"
    if [[ "${PUSH_TO_MAIN:-}" == "true" ]]; then
        PUSH_EXPORTS="$PUSH_EXPORTS,PUSH_TO_MAIN=true"
    fi
    if [[ -n "${GIT_REMOTE:-}" ]]; then
        PUSH_EXPORTS="$PUSH_EXPORTS,GIT_REMOTE=$GIT_REMOTE"
    fi
    if [[ -n "${NOTIFY_EMAIL:-}" ]]; then
        PUSH_EXPORTS="$PUSH_EXPORTS,NOTIFY_EMAIL=$NOTIFY_EMAIL"
    fi
    PUSH_JID=$(sbatch --parsable --dependency=afterany:"$PARENT_JOBS" \
        --export="$PUSH_EXPORTS" \
        cluster/99_push_results.slurm)
    echo "  (D) autopush             JID=$PUSH_JID  (afterany:$PARENT_JOBS)"
else
    echo "  (D) autopush             SKIPPED (no parent jobs to push for)"
fi

echo ""
echo "Submitted. Monitor via:"
echo "  squeue -u \$USER"
echo "  tail -f cluster/logs/*_${SMOKE_JID:-?}.out"
