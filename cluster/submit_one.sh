#!/usr/bin/env bash
# =============================================================================
# Single-job submit + chained auto-push (Pattern A debug-time variant)
# =============================================================================
# Submits one fitting SLURM and chains cluster/99_push_results.slurm via
# --dependency=afterany so results push back even on crash / timeout / cancel.
# Use for debug runs of one job — gives per-job pushes without running the
# full pipeline orchestrator.
#
# Why this and not autopush.sh: per-job EXIT-trap pushes (Pattern B) race
# on shared NFS staging when fan-out jobs run in parallel and silently
# drop files via rebase stash-pop conflicts (rlwm 1c37d3f / bef42fc).
# This wrapper keeps Pattern A's single-push-per-submission guarantee.
#
# Usage:
#   bash cluster/submit_one.sh cluster/<job>.slurm
#   bash cluster/submit_one.sh cluster/<job>.slurm --export=MODEL=foo
#   NOTIFY_EMAIL=adam.manoogian@monash.edu \
#       bash cluster/submit_one.sh cluster/<job>.slurm
#   GIT_REMOTE=ngithub bash cluster/submit_one.sh cluster/<job>.slurm
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

JOB="${1:?Usage: submit_one.sh <cluster/job.slurm> [extra sbatch args...]}"
shift

[[ -f "$JOB" ]] || { echo "ERROR: not found: $JOB" >&2; exit 2; }
[[ -f cluster/99_push_results.slurm ]] || {
    echo "ERROR: cluster/99_push_results.slurm missing — run /slurm-autopush" >&2
    exit 2
}

NOTIFY_EMAIL="${NOTIFY_EMAIL:-}"
GIT_REMOTE="${GIT_REMOTE:-origin}"

# CRLF strip — Windows checkouts, Pattern A pre-flight (slurm-autopush skill)
sed -i 's/\r$//' "$JOB" cluster/99_push_results.slurm 2>/dev/null || true

JID=$(sbatch --parsable "$@" "$JOB")
echo "Submitted fit:  JID=$JID  ($JOB)"

PUSH_JID=$(sbatch --parsable \
    --dependency=afterany:$JID \
    --export="ALL,PARENT_JOBS=$JID,NOTIFY_EMAIL=${NOTIFY_EMAIL},GIT_REMOTE=${GIT_REMOTE}" \
    cluster/99_push_results.slurm)
echo "Submitted push: JID=$PUSH_JID  (afterany:$JID)"
echo ""
echo "Watch logs: tail -f cluster/logs/*${JID}* logs/*${JID}* 2>/dev/null"
echo "Cancel:     scancel $JID $PUSH_JID"
