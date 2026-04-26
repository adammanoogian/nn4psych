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
# Auto-push results after this single job (recommended pattern):
#   sed -i 's/\r$//' cluster/*.slurm cluster/*.sh                          # CRLF strip
#   FIT_JID=$(sbatch --parsable cluster/run_circuit_ensemble.sh)
#   sbatch --parsable --dependency=afterany:${FIT_JID} \
#       --export=ALL,PARENT_JOBS="${FIT_JID}" \
#       cluster/99_push_results.slurm
#   # Use afterany (not afterok) so push fires on failure too.
#   # Add NOTIFY_EMAIL=adam.manoogian@monash.edu to the push --export for email.
#   # See cluster/run_n_latent_sweep.sh for the multi-job (sweep) variant
#   # — it wires the same push job automatically across all 4 fitting jobs.
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
#SBATCH --time=04:00:00
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
INCLUDE_OUTPUT_LOSS=${INCLUDE_OUTPUT_LOSS:-1}  # Set to 0 to fit hidden states only
FORCE_RETRAIN=${FORCE_RETRAIN:-0}  # Set to 1 to regenerate training data
FORCE_RECOLLECT=${FORCE_RECOLLECT:-0}  # Set to 1 to re-collect circuit data (keep model)
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-}   # If set, write to output/circuit_analysis/${OUTPUT_SUBDIR}/ instead of root
MASKED=${MASKED:-0}  # Set to 1 for masked-loss fitting (Gap 1, Plan 03-05)

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
# Start GPU monitoring in background (logs every 10s)
if command -v nvidia-smi &>/dev/null && [[ -n "$SLURM_JOB_ID" ]]; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv -l 10 > "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv" 2>/dev/null &
    GPU_MON_PID=$!
    echo "GPU monitoring started (PID $GPU_MON_PID) -> cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv"
fi
echo ""

# Generate data if not present (training + collection)
mkdir -p cluster/logs output/circuit_analysis data/processed/rnn_behav

# Serialize data regen across concurrent jobs via flock. Without this, parallel
# sbatch submissions race on circuit_data.npz — either with FORCE_RECOLLECT
# deletions or on the regen write — and FileNotFoundError crashes the loser
# (see job 54923593). FORCE_* deletions are also moved inside the lock so they
# cannot clobber a file another job is currently reading.
LOCKFILE="data/processed/rnn_behav/.regen.lock"
(
    flock -x 200

    if [[ "$FORCE_RECOLLECT" == "1" ]] && [[ -f "data/processed/rnn_behav/circuit_data.npz" ]]; then
        echo "FORCE_RECOLLECT=1: removing old circuit data (keeping trained model)..."
        rm -f data/processed/rnn_behav/circuit_data.npz data/processed/rnn_behav/circuit_data_metadata.json
    fi

    if [[ "$FORCE_RETRAIN" == "1" ]] && [[ -f "data/processed/rnn_behav/circuit_data.npz" ]]; then
        echo "FORCE_RETRAIN=1: removing old data to regenerate..."
        rm -f data/processed/rnn_behav/circuit_data.npz data/processed/rnn_behav/model_context_dm_dual.pth
    fi

    if [[ ! -f "data/processed/rnn_behav/circuit_data.npz" ]]; then
        echo "circuit_data.npz not found. Running training + collection (lock held)..."
        echo ""

        # Step 1: Train dual-modality ContinuousActorCritic (150 epochs, closer to paper)
        python -u scripts/training/train_context_dm.py \
            --both_modalities \
            --epochs 150 --trials 200 \
            --hidden_dim 64 --seed 42 \
            --skip_extraction || { echo "ERROR: Training failed"; exit 1; }

        # Step 2: Collect circuit data (u, z, y) — 500 per context = 1000 total
        python -u -c "
import sys; sys.path.insert(0, '.')
import torch
from nn4psych.models.continuous_rnn import ContinuousActorCritic
from nn4psych.analysis.circuit_inference import collect_circuit_data, save_circuit_data

model = ContinuousActorCritic(7, 64, 3, alpha=0.2, sigma_rec=0.15, gain=0.9)
model.load_state_dict(torch.load('data/processed/rnn_behav/model_context_dm_dual.pth', map_location='cpu'))
data = collect_circuit_data(model, n_trials_per_context=500, max_steps=75)
save_circuit_data(data, 'data/processed/rnn_behav')
print(f'Circuit data: u={data[\"u\"].shape}, z={data[\"z\"].shape}, y={data[\"y\"].shape}')
" || { echo "ERROR: Data collection failed"; exit 1; }
        echo ""
    else
        echo "circuit_data.npz present, skipping regen (another job wrote it or it was pre-existing)."
    fi
) 200>"$LOCKFILE"

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
include_output_loss = bool(int(os.environ.get('INCLUDE_OUTPUT_LOSS', 1)))

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

# MASKED=1 assertion (Plan 03-05): if the masked-loss path is requested,
# circuit_data.npz MUST contain task_active_mask. If missing, the cluster
# checkout is stale and the fallback regen (in the flock guard above) would
# have re-collected WITHOUT the mask key — silently producing wrong results.
masked_mode = bool(int(os.environ.get('MASKED', 0)))
if masked_mode:
    if 'task_active_mask' not in data.files:
        raise KeyError(
            "MASKED=1 set but circuit_data.npz lacks 'task_active_mask' key — "
            "cluster checkout is stale; pull latest main:\n"
            "  cd <project_root> && git pull origin main\n"
            "Expected after Plan 03-05 Task 1D commits circuit_data.npz with "
            "task_active_mask (1000, 75) bool."
        )
    task_active_mask = data['task_active_mask']
    active_per_trial = task_active_mask.sum(axis=1)
    print(
        f"  task_active_mask: mean_active={float(active_per_trial.mean()):.1f}, "
        f"min={int(active_per_trial.min())}, max={int(active_per_trial.max())} steps"
    )
else:
    task_active_mask = None

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

print(f"  include_output_loss={include_output_loss}, l_y={l_y}, n_latent={n_latent}")
print(f"  z range: [{z.min():.3f}, {z.max():.3f}] (should be [0,1] for softmax beliefs)")

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
    include_output_loss=include_output_loss,
    task_active_mask=task_active_mask,  # None when MASKED=0 (Wave A path unchanged)
)

elapsed = time.time() - start_time
print(f"\nEnsemble complete in {elapsed/60:.1f} minutes")
print(f"Best init: {result['best_init_idx']} (nmse_y={result['best_nmse_y']:.4f})")

# Save best model (per-rank subdir support via OUTPUT_SUBDIR env var)
_output_subdir = os.environ.get('OUTPUT_SUBDIR', '')
_output_root = (
    os.path.join('output/circuit_analysis', _output_subdir)
    if _output_subdir
    else 'output/circuit_analysis'
)
os.makedirs(_output_root, exist_ok=True)
torch.save(
    result['best_model'].state_dict(),
    os.path.join(_output_root, 'best_latent_circuit.pt'),
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
    'plan': '05' if masked_mode else '02',
    'n_inits': n_inits,
    'n_latent': n_latent,
    'epochs': epochs,
    'device': device,
    'elapsed_minutes': round(elapsed / 60, 1),
    'best_init_idx': int(result['best_init_idx']),
    'best_nmse_y': float(result['best_nmse_y']),
    'best_nmse_y_full': float(result['best_nmse_y_full']),
    'best_mse_z': float(result['best_mse_z']),
    'masked': bool(masked_mode),
    'invariant_subspace_corr': float(val['invariant_subspace_corr']),
    'invariant_subspace_pass': bool(val['invariant_subspace_pass']),
    'invariant_subspace_threshold': 0.85,
    'activity_r2_full_space': float(val['activity_r2_full_space'])
        if val['activity_r2_full_space'] is not None else None,
    'activity_r2_latent_space': float(val['activity_r2_latent_space'])
        if val['activity_r2_latent_space'] is not None else None,
    'trial_avg_r2_full_space': float(val['trial_avg_r2_full_space'])
        if val['trial_avg_r2_full_space'] is not None else None,
    'trial_avg_r2_latent_space': float(val['trial_avg_r2_latent_space'])
        if val['trial_avg_r2_latent_space'] is not None else None,
    'trial_avg_r2_by_condition': {
        int(k): float(v) for k, v in val['trial_avg_r2_by_condition'].items()
    } if val['trial_avg_r2_by_condition'] is not None else None,
    'nmse_y': float(val['nmse_y']),
    'nmse_q': float(val['nmse_q']),
    'mse_z': float(val['mse_z']),
    'data_shape': {
        'n_trials': int(u.shape[0]),
        'T': int(u.shape[1]),
        'N': int(y.shape[2]),
    },
    'status': 'pass' if val['invariant_subspace_pass'] else 'soft-fail',
    'output_subdir': _output_subdir,
}
with open(os.path.join(_output_root, 'validation_results.json'), 'w') as f:
    json.dump(val_report, f, indent=2)

diagnostics = {
    'all_nmse_y': [float(v) for v in result['all_nmse_y']],
    'all_nmse_y_full': [float(v) for v in result['all_nmse_y_full']],
    'all_mse_z': [float(v) for v in result['all_mse_z']],
    'masked': bool(masked_mode),
    'convergence_stats': {
        'mean_nmse_y': float(np.mean(result['all_nmse_y'])),
        'std_nmse_y': float(np.std(result['all_nmse_y'])),
        'min_nmse_y': float(np.min(result['all_nmse_y'])),
        'max_nmse_y': float(np.max(result['all_nmse_y'])),
    },
}
with open(os.path.join(_output_root, 'ensemble_diagnostics.json'), 'w') as f:
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

# Stop GPU monitoring and print summary
if [[ -n "${GPU_MON_PID:-}" ]]; then
    kill $GPU_MON_PID 2>/dev/null
    wait $GPU_MON_PID 2>/dev/null

    echo ""
    echo "=== GPU UTILIZATION SUMMARY ==="
    if [[ -f "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv" ]]; then
        # Print header + stats
        head -1 "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv"
        echo "Samples: $(wc -l < "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv")"
        # Average GPU util (column 2, values like "85 %")
        awk -F', ' 'NR>1 {sum+=$2+0; n++} END {if(n>0) printf "Avg GPU util: %.1f%%\n", sum/n; else print "No GPU data"}' \
            "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv"
        # Average memory used (column 4, values like "2048 MiB")
        awk -F', ' 'NR>1 {sum+=$4+0; n++} END {if(n>0) printf "Avg memory: %.0f MiB\n", sum/n}' \
            "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv"
        # Peak GPU util
        awk -F', ' 'NR>1 {v=$2+0; if(v>max) max=v} END {printf "Peak GPU util: %.0f%%\n", max}' \
            "cluster/logs/gpu_stats_${SLURM_JOB_ID}.csv"
    fi
    echo "================================"
fi

echo ""
echo "Job complete at $(date). Exit code: $?"
