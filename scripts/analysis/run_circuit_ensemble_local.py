"""Local CPU runner for the 03-02 latent circuit ensemble.

Mirrors the inline Python block in cluster/run_circuit_ensemble.sh so local
and cluster runs produce identical artifacts in output/circuit_analysis/.
Reads configuration from env vars (N_INITS, EPOCHS, N_LATENT, LR, L_Y,
WEIGHT_DECAY, INCLUDE_OUTPUT_LOSS) for parity with the SLURM script.

Exit codes: 0 success, 1 on any failure.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nn4psych.analysis.circuit_inference import (  # noqa: E402
    fit_latent_circuit_ensemble,
    validate_latent_circuit,
)
from nn4psych.models.continuous_rnn import ContinuousActorCritic  # noqa: E402


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    n_inits = int(os.environ.get("N_INITS", 100))
    epochs = int(os.environ.get("EPOCHS", 500))
    n_latent = int(os.environ.get("N_LATENT", 8))
    lr = float(os.environ.get("LR", 0.02))
    l_y = float(os.environ.get("L_Y", 1.0))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.001))
    include_output_loss = bool(int(os.environ.get("INCLUDE_OUTPUT_LOSS", 1)))

    data_path = ROOT / "data" / "processed" / "rnn_behav" / "circuit_data.npz"
    model_path = ROOT / "data" / "processed" / "rnn_behav" / "model_context_dm_dual.pth"
    out_dir = ROOT / "output" / "circuit_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading circuit data...", flush=True)
    data = np.load(data_path)
    u, z, y = data["u"], data["z"], data["y"]
    labels = {
        "modality_context": data["labels_modality_context"],
        "coherence_sign": data["labels_coherence_sign"],
        "correct_action": data["labels_correct_action"],
    }
    print(f"  u={u.shape}, z={z.shape}, y={y.shape}", flush=True)
    print(f"  z range: [{z.min():.3f}, {z.max():.3f}]", flush=True)

    print("Loading trained model...", flush=True)
    model = ContinuousActorCritic(
        input_dim=u.shape[2],
        hidden_dim=y.shape[2],
        action_dim=z.shape[2],
        alpha=0.2,
        sigma_rec=0.15,
        gain=0.9,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    W_rec = model.W_hh.weight.data.detach().numpy()
    W_in = model.W_ih.weight.data.detach().numpy()

    print(
        f"\nStarting {n_inits}-init ensemble ({epochs} epochs each, device={device})...",
        flush=True,
    )
    print(
        f"  include_output_loss={include_output_loss}, l_y={l_y}, n_latent={n_latent}",
        flush=True,
    )

    start = time.time()
    result = fit_latent_circuit_ensemble(
        u,
        z,
        y,
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
    )
    elapsed = time.time() - start
    print(f"\nEnsemble complete in {elapsed / 60:.1f} minutes", flush=True)
    print(
        f"Best init: {result['best_init_idx']} (nmse_y={result['best_nmse_y']:.4f})",
        flush=True,
    )

    torch.save(result["best_model"].state_dict(), out_dir / "best_latent_circuit.pt")

    print("\nRunning validation...", flush=True)
    val = validate_latent_circuit(
        result["best_model"],
        u,
        z,
        y,
        W_rec,
        W_in,
        labels=labels,
        invariant_threshold=0.85,
        device=device,
    )

    report = {
        "phase": "03-latent-circuit-inference",
        "plan": "02",
        "run_location": "local-cpu" if device == "cpu" else f"local-{device}",
        "n_inits": n_inits,
        "n_latent": n_latent,
        "epochs": epochs,
        "device": device,
        "elapsed_minutes": round(elapsed / 60, 1),
        "best_init_idx": result["best_init_idx"],
        "best_nmse_y": result["best_nmse_y"],
        "best_mse_z": result["best_mse_z"],
        "invariant_subspace_corr": val["invariant_subspace_corr"],
        "invariant_subspace_pass": val["invariant_subspace_pass"],
        "invariant_subspace_threshold": 0.85,
        "activity_r2_full_space": val["activity_r2_full_space"],
        "activity_r2_latent_space": val["activity_r2_latent_space"],
        "trial_avg_r2_full_space": val["trial_avg_r2_full_space"],
        "trial_avg_r2_latent_space": val["trial_avg_r2_latent_space"],
        "trial_avg_r2_by_condition": val["trial_avg_r2_by_condition"],
        "nmse_y": val["nmse_y"],
        "nmse_q": val["nmse_q"],
        "mse_z": val["mse_z"],
        "data_shape": {
            "n_trials": int(u.shape[0]),
            "T": int(u.shape[1]),
            "N": int(y.shape[2]),
        },
        "status": "pass" if val["invariant_subspace_pass"] else "soft-fail",
    }
    with open(out_dir / "validation_results.json", "w") as f:
        json.dump(report, f, indent=2)

    diagnostics = {
        "all_nmse_y": result["all_nmse_y"],
        "all_mse_z": result["all_mse_z"],
        "convergence_stats": {
            "mean_nmse_y": float(np.mean(result["all_nmse_y"])),
            "std_nmse_y": float(np.std(result["all_nmse_y"])),
            "min_nmse_y": float(np.min(result["all_nmse_y"])),
            "max_nmse_y": float(np.max(result["all_nmse_y"])),
        },
    }
    with open(out_dir / "ensemble_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"  nmse_y (best):           {report['best_nmse_y']:.4f}", flush=True)
    status_label = "PASS" if report["invariant_subspace_pass"] else "SOFT-FAIL"
    print(
        f"  Invariant subspace corr: {report['invariant_subspace_corr']:.4f} {status_label}",
        flush=True,
    )
    print(f"  Activity R2 (full):      {report['activity_r2_full_space']:.4f}", flush=True)
    print(f"  Activity R2 (latent):    {report['activity_r2_latent_space']:.4f}", flush=True)
    if report["trial_avg_r2_full_space"] is not None:
        print(f"  Trial-avg R2 (full):     {report['trial_avg_r2_full_space']:.4f}", flush=True)
    print(f"  Status:                  {report['status']}", flush=True)
    print(f"  Time:                    {report['elapsed_minutes']} min", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
