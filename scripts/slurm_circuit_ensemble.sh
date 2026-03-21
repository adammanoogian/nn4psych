#!/bin/bash
#SBATCH -J circuit_ensemble
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -p seas_gpu
#SBATCH --mem=8G
#SBATCH -o logs/circuit_ensemble_%j.out
#SBATCH -e logs/circuit_ensemble_%j.err

# Latent circuit inference: 100-init LatentNet ensemble fitting on GPU
# Expects circuit_data.npz to already exist from local data collection
#
# Usage:
#   sbatch scripts/slurm_circuit_ensemble.sh
#
# Or with custom parameters:
#   sbatch scripts/slurm_circuit_ensemble.sh --n_inits 100 --epochs 500

eval "$(conda shell.bash hook)"
conda activate pytorch

mkdir -p logs output/circuit_analysis

# Default parameters (can be overridden via command line args passed to sbatch)
N_INITS=${N_INITS:-100}
EPOCHS=${EPOCHS:-500}
N_LATENT=${N_LATENT:-8}
LR=${LR:-0.02}
L_Y=${L_Y:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}

echo "=== Latent Circuit Ensemble Fitting ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "n_inits: $N_INITS"
echo "epochs: $EPOCHS"
echo "n_latent: $N_LATENT"
echo "========================================="

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
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# Read env vars for parameters
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

# Load trained model for W_rec
print("Loading trained model for W_rec extraction...")
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
print(f"\nStarting {n_inits}-init ensemble ({epochs} epochs each)...")
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
print("Saved: output/circuit_analysis/best_latent_circuit.pt")

# Run validation
print("\nRunning validation...")
val = validate_latent_circuit(
    result['best_model'], u, z, y, W_rec, W_in,
    labels=labels,
    invariant_threshold=0.85,
    device=device,
)

# Save validation results
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
print("Saved: output/circuit_analysis/validation_results.json")

# Save ensemble diagnostics
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
print("Saved: output/circuit_analysis/ensemble_diagnostics.json")

# Print summary
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  nmse_y (best):           {val_report['best_nmse_y']:.4f}")
print(f"  Invariant subspace corr: {val_report['invariant_subspace_corr']:.4f} {'PASS' if val_report['invariant_subspace_pass'] else 'SOFT-FAIL'}")
print(f"  Activity R2 (full):      {val_report['activity_r2_full_space']:.4f}")
print(f"  Activity R2 (latent):    {val_report['activity_r2_latent_space']:.4f}")
if val_report['trial_avg_r2_full_space'] is not None:
    print(f"  Trial-avg R2 (full):     {val_report['trial_avg_r2_full_space']:.4f}")
    print(f"  Trial-avg by condition:  {val_report['trial_avg_r2_by_condition']}")
print(f"  Status:                  {val_report['status']}")
print(f"  Time:                    {val_report['elapsed_minutes']} min")
print("=" * 60)
PYTHON_SCRIPT

echo "Job complete. Exit code: $?"
