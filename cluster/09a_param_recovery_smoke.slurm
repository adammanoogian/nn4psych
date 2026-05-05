#!/bin/bash
# =============================================================================
# SLURM: Phase 4 Stage 09a — Bigger smoke parameter recovery (CPU)
# =============================================================================
# Re-runs scripts/data_pipeline/09a_param_recovery.py at smoke-grade settings
# with more datasets and trials than the local v2 smoke, to confirm the model
# is identifiable for cognitive parameters (H, LW, log_UU) when given more
# statistical power than N=4 × 100 trials × 500/500/2.
#
# Settings (tuned bigger than local smoke, smaller than full recovery):
#   N synthetic datasets:    8
#   Trials per condition:    200
#   MCMC config:             1000 warmup × 1000 samples × 2 chains
#   Output dir:              data/processed/bayesian/param_recovery_smoke_v3
#
# Wall time on cluster CPU (estimate):
#   16 fits × ~3-5 min/fit ≈ 60-90 min total
#
# Usage:
#   sbatch cluster/09a_param_recovery_smoke.slurm
# =============================================================================

#SBATCH --job-name=rbo_smoke_v3
#SBATCH --output=cluster/logs/rbo_smoke_v3_%j.out
#SBATCH --error=cluster/logs/rbo_smoke_v3_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

set -e

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "RBO smoke recovery v3 — SLURM Job"
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
mkdir -p cluster/logs data/processed/bayesian/param_recovery_smoke_v3

# =============================================================================
# Run smoke recovery
# =============================================================================
python -u scripts/data_pipeline/09a_param_recovery.py \
    --n_synthetic 8 \
    --n_trials_per_condition 200 \
    --num_warmup 1000 \
    --num_samples 1000 \
    --num_chains 2 \
    --output_dir data/processed/bayesian/param_recovery_smoke_v3

PYTHON_EXIT=$?

echo ""
echo "Job complete at $(date). Exit code: $PYTHON_EXIT"
exit $PYTHON_EXIT
