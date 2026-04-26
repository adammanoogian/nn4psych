#!/bin/bash
# =============================================================================
# SLURM: Per-Context Latent Circuit Fit (single modality_context)
# =============================================================================
# Fits a 100-init LatentNet ensemble on ONE modality_context slice of
# circuit_data.npz. Submit via cluster/run_per_context_fits.sh which launches
# both context 0 and context 1 in parallel and wires autopush.
#
# Usage (direct sbatch — normally invoked by run_per_context_fits.sh):
#   sbatch --export=ALL,CONTEXT_ID=0 cluster/run_per_context_one.slurm
#   sbatch --export=ALL,CONTEXT_ID=1 cluster/run_per_context_one.slurm
#
# Performance (approximate, ~500 trials x T=75):
#   - GPU: ~95 min (100 inits x 500 epochs) — matches Wave A wall time
#
# Part of: Plan 03-06 (Gap 3 per-context diagnostic)
# =============================================================================

#SBATCH --job-name=circuit_per_ctx0
#SBATCH --output=cluster/logs/circuit_per_ctx%a_%j.out
#SBATCH --error=cluster/logs/circuit_per_ctx%a_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Configuration (override via --export=VAR=val)
# =============================================================================
CONTEXT_ID=${CONTEXT_ID:-0}
N_LATENT=${N_LATENT:-12}
N_INITS=${N_INITS:-100}
EPOCHS=${EPOCHS:-500}

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Per-Context Latent Circuit Fit — SLURM Job"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start:      $(date)"
echo "context_id: $CONTEXT_ID"
echo "n_latent:   $N_LATENT"
echo "n_inits:    $N_INITS"
echo "epochs:     $EPOCHS"
echo "============================================================"

# Activate project environment
conda activate actinf-py-scripts 2>/dev/null || conda activate nn4psych || {
    echo "ERROR: No conda env found (tried actinf-py-scripts, nn4psych)"
    echo "Run: bash cluster/setup_env.sh"
    exit 1
}

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

# =============================================================================
# Verify GPU Access
# =============================================================================
echo ""
echo "Verifying GPU..."
python -c "
import torch
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU device: {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('  WARNING: No GPU detected. Falling back to CPU.')
"

# Start GPU monitoring in background
if command -v nvidia-smi &>/dev/null && [[ -n "$SLURM_JOB_ID" ]]; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv -l 10 > "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv" 2>/dev/null &
    GPU_MON_PID=$!
    echo "GPU monitoring started (PID $GPU_MON_PID)"
fi
echo ""

mkdir -p cluster/logs "output/circuit_analysis/per_context/context_${CONTEXT_ID}"

# =============================================================================
# Run Per-Context Fitting
# =============================================================================
python -u scripts/analysis/fit_per_context_latent_circuits.py \
    --context "$CONTEXT_ID" \
    --n_latent "$N_LATENT" \
    --n_inits "$N_INITS" \
    --epochs "$EPOCHS" \
    --output_dir "output/circuit_analysis/per_context/context_${CONTEXT_ID}/" \
    --device cuda

PYTHON_EXIT=$?

# Stop GPU monitoring
if [[ -n "${GPU_MON_PID:-}" ]]; then
    kill $GPU_MON_PID 2>/dev/null
    wait $GPU_MON_PID 2>/dev/null

    echo ""
    echo "=== GPU UTILIZATION SUMMARY ==="
    if [[ -f "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv" ]]; then
        head -1 "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv"
        echo "Samples: $(wc -l < "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv")"
        awk -F', ' 'NR>1 {sum+=$2+0; n++} END {if(n>0) printf "Avg GPU util: %.1f%%\n", sum/n; else print "No GPU data"}' \
            "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv"
        awk -F', ' 'NR>1 {sum+=$4+0; n++} END {if(n>0) printf "Avg memory: %.0f MiB\n", sum/n}' \
            "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv"
    fi
    echo "================================"
fi

echo ""
echo "Job complete at $(date). Exit code: $PYTHON_EXIT"
exit $PYTHON_EXIT
