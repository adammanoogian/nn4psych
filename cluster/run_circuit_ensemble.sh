#!/bin/bash
# =============================================================================
# SLURM: GPU-Accelerated Latent Circuit Ensemble Fitting
# =============================================================================
# Runs 100-initialization LatentNet ensemble fitting on GPU.
# Expects circuit_data.npz to exist from local data collection (Phase 3, Plan 01).
#
# Usage:
#   sbatch cluster/run_circuit_ensemble.sh
#   sbatch --export=N_INITS=50,EPOCHS=300 cluster/run_circuit_ensemble.sh   # Quick test
#
# Performance (approximate, 600 trials x T=75):
#   - CPU:  ~2-3 hours (100 inits x 500 epochs)
#   - GPU:  ~15-30 minutes
#
# Derived from: project_utils/templates/slurm_gpu_TEMPLATE.slurm
# =============================================================================

#SBATCH --job-name=circuit_ensemble
#SBATCH --output=cluster/logs/circuit_ensemble_%j.out
#SBATCH --error=cluster/logs/circuit_ensemble_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Configuration (override via --export=VAR=val)
# =============================================================================
N_INITS=${N_INITS:-100}
EPOCHS=${EPOCHS:-500}
N_LATENT=${N_LATENT:-8}
LR=${LR:-0.02}
L_Y=${L_Y:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}

# =============================================================================
# Environment Setup
# =============================================================================
# Load miniforge3 (M3 convention). Falls back to conda hook on other clusters.
module load miniforge3 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Latent Circuit Ensemble Fitting — SLURM Job"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start:      $(date)"
echo "n_inits:    $N_INITS"
echo "epochs:     $EPOCHS"
echo "n_latent:   $N_LATENT"
echo "============================================================"

# Activate project environment
conda activate actinf-py-scripts 2>/dev/null || conda activate nn4psych || {
    echo "ERROR: No conda env found (tried actinf-py-scripts, nn4psych)"
    echo "Run: bash cluster/setup_env.sh"
    exit 1
}

# Add project root to PYTHONPATH so nn4psych and envs are importable
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
echo ""

# Generate data if not present (training + collection)
mkdir -p cluster/logs output/circuit_analysis data/processed/rnn_behav

if [[ ! -f "data/processed/rnn_behav/circuit_data.npz" ]]; then
    echo "circuit_data.npz not found. Running training + collection..."
    echo ""

    # Step 1: Train dual-modality ContinuousActorCritic
    python -u scripts/training/train_context_dm.py \
        --both_modalities \
        --epochs 50 --trials 100 \
        --hidden_dim 64 --seed 42 \
        --skip_extraction || { echo "ERROR: Training failed"; exit 1; }

    # Step 2: Collect circuit data (u, z, y) with proper params
    python -u -c "
import sys; sys.path.insert(0, '.')
import torch
from nn4psych.models.continuous_rnn import ContinuousActorCritic
from nn4psych.analysis.circuit_inference import collect_circuit_data, save_circuit_data

model = ContinuousActorCritic(7, 64, 3, alpha=0.2, sigma_rec=0.15, gain=0.9)
model.load_state_dict(torch.load('data/processed/rnn_behav/model_context_dm_dual.pth', map_location='cpu'))
data = collect_circuit_data(model, n_trials_per_context=300, max_steps=75)
save_circuit_data(data, 'data/processed/rnn_behav')
print(f'Circuit data: u={data[\"u\"].shape}, z={data[\"z\"].shape}, y={data[\"y\"].shape}')
" || { echo "ERROR: Data collection failed"; exit 1; }
    echo ""
fi

# =============================================================================
# Run Ensemble Fitting + Validation
# =============================================================================
python -u - <<'PYTHON_SCRIPT'
import sys, os, json, time
sys.path.insert(0, '.')

import numpy as np
import torch
from nn4psych.models.continuous_rnn import ContinuousActorCritic
from nn4psych.analysis.circuit_inference import (
    fit_latent_circuit_ensemble,
    validate_latent_circuit,
)

# Auto-detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Read parameters from environment
n_inits = int(os.environ.get('N_INITS', 100))
epochs = int(os.environ.get('EPOCHS', 500))
n_latent = int(os.environ.get('N_LATENT', 8))
lr = float(os.environ.get('LR', 0.02))
l_y = float(os.environ.get('L_Y', 1.0))
weight_decay = float(os.environ.get('WEIGHT_DECAY', 0.001))

# Load circuit data
print("\nLoading circuit data...")
data = np.load('data/processed/rnn_behav/circuit_data.npz')
u, z, y = data['u'], data['z'], data['y']
labels = {
    'modality_context': data['labels_modality_context'],
    'coherence_sign': data['labels_coherence_sign'],
    'correct_action': data['labels_correct_action'],
}
print(f"  u={u.shape}, z={z.shape}, y={y.shape}")

# Load trained model for W_rec extraction
print("Loading trained model...")
model = ContinuousActorCritic(
    input_dim=u.shape[2], hidden_dim=y.shape[2], action_dim=z.shape[2],
    alpha=0.2, sigma_rec=0.15, gain=0.9,
)
model.load_state_dict(torch.load(
    'data/processed/rnn_behav/model_context_dm_dual.pth',
    map_location='cpu',
))
W_rec = model.W_hh.weight.data.detach().numpy()
W_in = model.W_ih.weight.data.detach().numpy()

# Run ensemble
print(f"\nStarting {n_inits}-init ensemble ({epochs} epochs each, device={device})...")
start_time = time.time()

result = fit_latent_circuit_ensemble(
    u, z, y,
    n_inits=n_inits,
    n_latent=n_latent,
    epochs=epochs,
    lr=lr,
    l_y=l_y,
    weight_decay=weight_decay,
    sigma_rec=0.15,
    device=device,
    verbose=True,
)

elapsed = time.time() - start_time
print(f"\nEnsemble complete in {elapsed/60:.1f} minutes")
print(f"Best init: {result['best_init_idx']} (nmse_y={result['best_nmse_y']:.4f})")

# Save best model
os.makedirs('output/circuit_analysis', exist_ok=True)
torch.save(
    result['best_model'].state_dict(),
    'output/circuit_analysis/best_latent_circuit.pt',
)

# Run validation
print("\nRunning validation...")
val = validate_latent_circuit(
    result['best_model'], u, z, y, W_rec, W_in,
    labels=labels,
    invariant_threshold=0.85,
    device=device,
)

# Save results
val_report = {
    'phase': '03-latent-circuit-inference',
    'plan': '02',
    'n_inits': n_inits,
    'n_latent': n_latent,
    'epochs': epochs,
    'device': device,
    'elapsed_minutes': round(elapsed / 60, 1),
    'best_init_idx': result['best_init_idx'],
    'best_nmse_y': result['best_nmse_y'],
    'best_mse_z': result['best_mse_z'],
    'invariant_subspace_corr': val['invariant_subspace_corr'],
    'invariant_subspace_pass': val['invariant_subspace_pass'],
    'invariant_subspace_threshold': 0.85,
    'activity_r2_full_space': val['activity_r2_full_space'],
    'activity_r2_latent_space': val['activity_r2_latent_space'],
    'trial_avg_r2_full_space': val['trial_avg_r2_full_space'],
    'trial_avg_r2_latent_space': val['trial_avg_r2_latent_space'],
    'trial_avg_r2_by_condition': val['trial_avg_r2_by_condition'],
    'nmse_y': val['nmse_y'],
    'nmse_q': val['nmse_q'],
    'mse_z': val['mse_z'],
    'data_shape': {
        'n_trials': int(u.shape[0]),
        'T': int(u.shape[1]),
        'N': int(y.shape[2]),
    },
    'status': 'pass' if val['invariant_subspace_pass'] else 'soft-fail',
}
with open('output/circuit_analysis/validation_results.json', 'w') as f:
    json.dump(val_report, f, indent=2)

diagnostics = {
    'all_nmse_y': result['all_nmse_y'],
    'all_mse_z': result['all_mse_z'],
    'convergence_stats': {
        'mean_nmse_y': float(np.mean(result['all_nmse_y'])),
        'std_nmse_y': float(np.std(result['all_nmse_y'])),
        'min_nmse_y': float(np.min(result['all_nmse_y'])),
        'max_nmse_y': float(np.max(result['all_nmse_y'])),
    },
}
with open('output/circuit_analysis/ensemble_diagnostics.json', 'w') as f:
    json.dump(diagnostics, f, indent=2)

# Summary
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  nmse_y (best):           {val_report['best_nmse_y']:.4f}")
print(f"  Invariant subspace corr: {val_report['invariant_subspace_corr']:.4f} {'PASS' if val_report['invariant_subspace_pass'] else 'SOFT-FAIL'}")
print(f"  Activity R2 (full):      {val_report['activity_r2_full_space']:.4f}")
print(f"  Activity R2 (latent):    {val_report['activity_r2_latent_space']:.4f}")
if val_report['trial_avg_r2_full_space'] is not None:
    print(f"  Trial-avg R2 (full):     {val_report['trial_avg_r2_full_space']:.4f}")
print(f"  Status:                  {val_report['status']}")
print(f"  Time:                    {val_report['elapsed_minutes']} min")
print("=" * 60)
PYTHON_SCRIPT

echo ""
echo "Job complete at $(date). Exit code: $?"
