#!/bin/bash
# =============================================================================
# SLURM ARRAY: Phase 4 Stage 09b — Per-subject RBO fits on Nassar 2021 data
# =============================================================================
# Fits the Reduced Bayesian Observer to all 134 subjects × 2 conditions = 268
# array tasks. Each task fits one (subject_idx, condition) pair using the
# full BAYES-06 grade MCMC settings (2000 warmup × 2000 samples × 4 chains).
#
# Array task indexing:
#   SLURM_ARRAY_TASK_ID in [0, 267]
#   subject_idx = task_id // 2          (0..133)
#   condition   = task_id % 2  → 0 = changepoint, 1 = oddball
#
# Wall time per task on cluster CPU:
#   ~5-15 min for ~190 trials per fit; 60 min ceiling allows headroom for
#   retry-on-failure (longer warmup) and slower nodes.
#
# Usage:
#   sbatch --array=0-267 cluster/09b_fit_human_subjects.slurm
#   # Or via orchestrator: bash cluster/run_phase4_bayesian.sh
# =============================================================================

#SBATCH --job-name=rbo_subj_fit
#SBATCH --output=cluster/logs/rbo_subj_fit_%A_%a.out
#SBATCH --error=cluster/logs/rbo_subj_fit_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp
#SBATCH --array=0-267

set -e

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

# Decompose array task ID into (subject_idx, condition)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SUBJECT_IDX=$((TASK_ID / 2))
COND_IDX=$((TASK_ID % 2))
if [[ "$COND_IDX" == "0" ]]; then
    CONDITION="changepoint"
else
    CONDITION="oddball"
fi

echo "============================================================"
echo "RBO per-subject fit — SLURM Array Task"
echo "============================================================"
echo "Job ID:        ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:          ${SLURMD_NODENAME:-$(hostname)}"
echo "Start:         $(date)"
echo "Array task:    $TASK_ID / 267"
echo "Subject idx:   $SUBJECT_IDX (0..133)"
echo "Condition:     $CONDITION"
echo "============================================================"

conda activate actinf-py-scripts 2>/dev/null || conda activate nn4psych || {
    echo "ERROR: No conda env found (tried actinf-py-scripts, nn4psych)"
    echo "Run: bash cluster/setup_env.sh"
    exit 1
}

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p cluster/logs data/processed/bayesian/human_validation/per_fit

# =============================================================================
# Run per-subject fit (full BAYES-06 settings)
# =============================================================================
python -u scripts/data_pipeline/09b_fit_human_subject.py \
    --subject_idx "$SUBJECT_IDX" \
    --condition "$CONDITION" \
    --num_warmup 2000 \
    --num_samples 2000 \
    --num_chains 4 \
    --output_dir data/processed/bayesian/human_validation

PYTHON_EXIT=$?

echo ""
echo "Task complete at $(date). Exit code: $PYTHON_EXIT"
exit $PYTHON_EXIT
