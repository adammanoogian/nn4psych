#!/bin/bash
# =============================================================================
# Environment Setup for nn4psych on SLURM Cluster (M3 / Monash compatible)
# =============================================================================
#
# Creates or updates the conda environment for RNN training, latent circuit
# inference, and Bayesian model fitting.
#
# Modelled on actinf_physics/cluster/setup_env.sh and M3 documentation:
#   https://docs.erc.monash.edu/Compute/HPC/M3/Software/Conda
#
# Usage:
#   bash cluster/setup_env.sh              # Create (or interactive if exists)
#   bash cluster/setup_env.sh --fresh      # Delete and recreate from scratch
#   bash cluster/setup_env.sh --gpu        # Add CUDA PyTorch after creation
#   bash cluster/setup_env.sh --check      # Verify existing environment
#
# Note: CUDA module is NOT required. PyTorch pip packages bundle their own
# CUDA runtime. GPU access works via the cluster's NVIDIA kernel driver.
# =============================================================================

set -e  # Exit on error (no -u to avoid PS1/unbound variable issues)

# Use shared env if actinf-py-scripts exists (saves ~4GB disk),
# otherwise create nn4psych-specific env
if conda env list 2>/dev/null | grep -q "actinf-py-scripts"; then
    ENV_NAME="actinf-py-scripts"
    echo "Found existing actinf-py-scripts env — will extend it (saves disk)"
else
    ENV_NAME="nn4psych"
fi
ENV_FILE="cluster/environment_cluster.yml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ─────────────────────────────────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────────────────────────────────

MODE="interactive"
while [[ $# -gt 0 ]]; do
    case $1 in
        --fresh|-f) MODE="fresh"; shift ;;
        --gpu|-g)   MODE="gpu"; shift ;;
        --check|-c) MODE="check"; shift ;;
        --help|-h)
            echo "Usage: bash cluster/setup_env.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fresh, -f     Delete and recreate environment"
            echo "  --gpu, -g       Install PyTorch with CUDA support"
            echo "  --check, -c     Verify existing environment"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "nn4psych Environment Setup (SLURM Cluster)"
echo "============================================================"
echo "Project: $PROJECT_ROOT"
echo "Mode: $MODE"
echo "Env name: $ENV_NAME"

# ─────────────────────────────────────────────────────────────────────────────
# Load miniforge3 (M3 convention — provides conda + mamba)
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "Loading modules..."
module load miniforge3 2>/dev/null || {
    echo "INFO: 'module load miniforge3' not available. Using system conda."
    # Fallback for non-module clusters
    eval "$(conda shell.bash hook)" 2>/dev/null || true
}

# ─────────────────────────────────────────────────────────────────────────────
# Configure conda for scratch storage (M3-specific)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ID="${M3_PROJECT:-}"
if [[ -z "$PROJECT_ID" ]]; then
    for dir in /scratch/*/$(whoami); do
        if [[ -d "$dir" ]]; then
            PROJECT_ID=$(basename "$(dirname "$dir")")
            break
        fi
    done
fi

if [[ -n "$PROJECT_ID" ]]; then
    CONDA_DIR="/scratch/$PROJECT_ID/$(whoami)/conda"
    echo "M3 project: $PROJECT_ID"
    echo "Conda dir:  $CONDA_DIR"
    mkdir -p "$CONDA_DIR/pkgs" "$CONDA_DIR/envs"
    conda config --prepend pkgs_dirs "$CONDA_DIR/pkgs" 2>/dev/null
    conda config --prepend envs_dirs "$CONDA_DIR/envs" 2>/dev/null
fi

# ─────────────────────────────────────────────────────────────────────────────
# Handle modes
# ─────────────────────────────────────────────────────────────────────────────

ENV_EXISTS=false
if conda env list | grep -q "$ENV_NAME"; then
    ENV_EXISTS=true
    echo "Environment '$ENV_NAME' already exists."
fi

if [[ "$MODE" == "check" ]]; then
    if [[ "$ENV_EXISTS" == false ]]; then
        echo "ERROR: Environment '$ENV_NAME' does not exist."
        exit 1
    fi
    echo "Running verification..."

elif [[ "$MODE" == "fresh" ]]; then
    if [[ "$ENV_EXISTS" == true ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    fi
    echo ""
    echo "Creating '$ENV_NAME' environment..."
    if command -v mamba &> /dev/null; then
        mamba env create -f "$PROJECT_ROOT/$ENV_FILE"
    else
        conda env create -f "$PROJECT_ROOT/$ENV_FILE"
    fi

elif [[ "$MODE" == "gpu" ]]; then
    if [[ "$ENV_EXISTS" == false ]]; then
        echo "ERROR: Environment '$ENV_NAME' does not exist. Create it first."
        exit 1
    fi
    echo ""
    echo "Installing PyTorch CUDA support..."
    conda activate "$ENV_NAME"
    # PyTorch bundles CUDA runtime — do NOT load cuda module
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    echo ""
    echo "Verifying GPU support..."
    python -c "
import torch
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU device: {torch.cuda.get_device_name(0)}')
    a = torch.randn(1000, 1000, device='cuda')
    b = a @ a
    print(f'  GPU matmul test: OK')
else:
    print('  INFO: No GPU on login node. Test on GPU node:')
    print('    srun --partition=gpu --gres=gpu:1 --time=00:10:00 --pty bash')
"
    echo "GPU setup complete!"
    exit 0

else
    # Interactive / default mode
    if [[ "$ENV_EXISTS" == true ]]; then
        echo ""
        echo "Installing nn4psych-specific packages into existing env..."
        echo "(This preserves all existing packages from other projects)"
        echo ""
        conda activate "$ENV_NAME"

        # Install only what nn4psych needs that may be missing
        pip install pyyaml tqdm psutil gymnasium \
            "neurogym @ git+https://github.com/neurogym/neurogym" 2>&1 | \
            grep -v "already satisfied" || true

        echo ""
        echo "Packages installed into existing '$ENV_NAME' env."
    else
        echo "Creating '$ENV_NAME' environment..."
        if command -v mamba &> /dev/null; then
            mamba env create -f "$PROJECT_ROOT/$ENV_FILE"
        else
            conda env create -f "$PROJECT_ROOT/$ENV_FILE"
        fi
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Verify installation
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"

conda activate "$ENV_NAME"

python -c "
import torch;       print(f'  torch:       {torch.__version__}')
import numpy;       print(f'  numpy:       {numpy.__version__}')
import scipy;       print(f'  scipy:       {scipy.__version__}')
import pandas;      print(f'  pandas:      {pandas.__version__}')
import matplotlib;  print(f'  matplotlib:  {matplotlib.__version__}')
try:
    import neurogym; print(f'  neurogym:    {neurogym.__version__}')
except ImportError:
    print('  neurogym:    NOT INSTALLED (optional)')
try:
    import jax;      print(f'  jax:         {jax.__version__}')
    import numpyro;  print(f'  numpyro:     {numpyro.__version__}')
except ImportError:
    print('  jax/numpyro: NOT INSTALLED (install with pip for Phase 4)')
print()
print('All core imports OK')
"

mkdir -p "$PROJECT_ROOT/cluster/logs"
mkdir -p "$PROJECT_ROOT/output/circuit_analysis"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Install GPU PyTorch (on GPU node or login):"
echo "     bash cluster/setup_env.sh --gpu"
echo ""
echo "  2. Test GPU access:"
echo "     srun --partition=gpu --gres=gpu:1 --time=00:10:00 --pty bash"
echo "     conda activate $ENV_NAME"
echo "     python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "  3. Run circuit ensemble:"
echo "     sbatch cluster/run_circuit_ensemble.sh"
echo ""
