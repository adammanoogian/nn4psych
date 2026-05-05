#!/bin/bash
# =============================================================================
# SLURM: Phase 4 Stage 09c — Aggregate per-subject RBO fits
# =============================================================================
# Reads all 268 per-fit JSONs produced by the 09b array job, builds a flat
# summary CSV, writes a markdown report, and saves diagnostic plots.
#
# Submitted with --dependency=afterok:$ARRAY_JID by the orchestrator so
# this only runs after every array task completes successfully.
#
# Wall time:
#   < 5 min on a single CPU.
#
# Usage:
#   sbatch --dependency=afterok:JOB cluster/09c_aggregate_human_fits.slurm
# =============================================================================

#SBATCH --job-name=rbo_subj_agg
#SBATCH --output=cluster/logs/rbo_subj_agg_%j.out
#SBATCH --error=cluster/logs/rbo_subj_agg_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=comp

set -e

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "RBO subject-fit aggregation — SLURM Job"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "Start:      $(date)"
echo "============================================================"

conda activate actinf-py-scripts 2>/dev/null || conda activate nn4psych || {
    echo "ERROR: No conda env found (tried actinf-py-scripts, nn4psych)"
    echo "Run: bash cluster/setup_env.sh"
    exit 1
}

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p cluster/logs figures/bayesian/human_validation

# =============================================================================
# Run aggregator
# =============================================================================
python -u scripts/data_pipeline/09c_aggregate_human_fits.py \
    --input_dir data/processed/bayesian/human_validation \
    --figures_dir figures/bayesian/human_validation

PYTHON_EXIT=$?

echo ""
echo "Job complete at $(date). Exit code: $PYTHON_EXIT"
exit $PYTHON_EXIT
