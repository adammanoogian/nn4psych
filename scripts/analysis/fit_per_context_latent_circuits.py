"""
Per-context latent circuit fitting driver.

Fits a LatentNet ensemble on a single modality_context slice of circuit_data.npz
(context 0 or 1). Used by the per-context diagnostic (Plan 03-06) to test whether
the SC-2 soft-fail (pooled corr=0.7833) is caused by the pooled fit averaging
over two structurally distinct low-rank subspaces.

Usage
-----
# Smoke test (local CPU, fast):
python scripts/analysis/fit_per_context_latent_circuits.py \
    --context 0 --n_latent 12 --n_inits 1 --epochs 5 \
    --output_dir output/circuit_analysis/per_context/smoke_test_ctx0/ \
    --device cpu

# Cluster full fit (submitted via cluster/run_per_context_fits.sh):
python scripts/analysis/fit_per_context_latent_circuits.py \
    --context 0 --n_latent 12 --n_inits 100 --epochs 500 \
    --output_dir output/circuit_analysis/per_context/context_0/ \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # MUST come before any pyplot import

import numpy as np  # noqa: E402, I001
import torch  # noqa: E402
from nn4psych.analysis.circuit_inference import (  # noqa: E402
    fit_latent_circuit_ensemble,
    validate_latent_circuit,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit LatentNet ensemble on a single modality_context slice."
    )
    p.add_argument(
        "--data_path",
        default="data/processed/rnn_behav/circuit_data.npz",
        help="Path to circuit_data.npz (default: %(default)s)",
    )
    p.add_argument(
        "--context",
        type=int,
        required=True,
        choices=[0, 1],
        help="modality_context to fit (0 or 1).",
    )
    p.add_argument(
        "--n_latent",
        type=int,
        default=12,
        help="LatentNet rank — matches Wave A chosen rank (default: %(default)s).",
    )
    p.add_argument(
        "--n_inits",
        type=int,
        default=100,
        help="Number of random initialisations (default: %(default)s).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Training epochs per initialisation (default: %(default)s).",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for outputs (default: "
            "output/circuit_analysis/per_context/context_<ctx>/)."
        ),
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device ('cuda' on cluster, 'cpu' locally; default: %(default)s).",
    )
    return p


# ---------------------------------------------------------------------------
# W_rec extraction
# ---------------------------------------------------------------------------

def _load_w_rec(checkpoint_path: str) -> np.ndarray:
    """Load W_rec from the canonical RNN checkpoint with defensive shape checks.

    Parameters
    ----------
    checkpoint_path : str
        Path to model_context_dm_dual.pth.

    Returns
    -------
    np.ndarray
        W_rec (W_hh.weight), shape (64, 64), float32.

    Raises
    ------
    AssertionError
        If checkpoint shapes do not match expected ContinuousActorCritic
        training configuration (hidden_dim=64, input_dim=7).
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Unwrap {'model': state_dict, ...} container if present
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    assert state_dict["W_hh.weight"].shape == (64, 64), (
        f"Expected W_hh.weight shape (64, 64), "
        f"got {tuple(state_dict['W_hh.weight'].shape)} — "
        "checkpoint is from a different training regime; abort and document."
    )
    assert state_dict["W_ih.weight"].shape == (64, 7), (
        f"Expected W_ih.weight shape (64, 7), "
        f"got {tuple(state_dict['W_ih.weight'].shape)} — "
        "checkpoint input_dim mismatch; abort and document."
    )

    w_rec: np.ndarray = state_dict["W_hh.weight"].detach().cpu().numpy()
    return w_rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for per-context latent circuit fitting."""
    parser = _build_parser()
    args = parser.parse_args()

    ctx: int = args.context
    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else f"output/circuit_analysis/per_context/context_{ctx}/"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load circuit data (READ-ONLY — do NOT modify circuit_data.npz)
    # ------------------------------------------------------------------
    print(f"Loading circuit data from {args.data_path} ...")
    npz = np.load(args.data_path)
    u_all: np.ndarray = npz["u"]
    z_all: np.ndarray = npz["z"]
    y_all: np.ndarray = npz["y"]
    mc_labels: np.ndarray = npz["labels_modality_context"]

    print(
        f"  Full dataset: u={u_all.shape}, z={z_all.shape}, y={y_all.shape}, "
        f"n_trials={u_all.shape[0]}"
    )

    # ------------------------------------------------------------------
    # Slice by modality_context
    # ------------------------------------------------------------------
    mask: np.ndarray = mc_labels == ctx
    u_ctx: np.ndarray = u_all[mask]
    z_ctx: np.ndarray = z_all[mask]
    y_ctx: np.ndarray = y_all[mask]

    n_trials_used: int = int(u_ctx.shape[0])
    print(f"  Context {ctx}: {n_trials_used} trials selected.")

    if n_trials_used < 100:
        raise ValueError(
            f"Expected >= 100 trials for context {ctx}, "
            f"got {n_trials_used}. "
            "Check that circuit_data.npz was generated with "
            "n_trials_per_context >= 100."
        )

    # ------------------------------------------------------------------
    # Load W_rec from canonical RNN checkpoint (defensive shape checks)
    # ------------------------------------------------------------------
    ckpt_path = "data/processed/rnn_behav/model_context_dm_dual.pth"
    print(f"Loading W_rec from {ckpt_path} ...")
    w_rec = _load_w_rec(ckpt_path)
    print(f"  W_rec shape: {w_rec.shape} — OK")

    # ------------------------------------------------------------------
    # Fit ensemble
    # Eval-procedure note: sigma_rec=0.15 default, no eval-mode override.
    # Corr is reported on the same noise realisation as training, matching
    # the cluster_same_seed_as_train procedure used for Wave A (pooled corr
    # 0.7833). This ensures per-context corrs are directly comparable to the
    # pooled baseline.
    # ------------------------------------------------------------------
    print(
        f"\nFitting {args.n_inits}-init ensemble "
        f"(context={ctx}, n_latent={args.n_latent}, "
        f"epochs={args.epochs}, device={args.device}) ..."
    )

    ensemble = fit_latent_circuit_ensemble(
        u_ctx,
        z_ctx,
        y_ctx,
        n_inits=args.n_inits,
        n_latent=args.n_latent,
        epochs=args.epochs,
        sigma_rec=0.15,   # matches Wave A default; no eval-mode override
        device=args.device,
        verbose=True,
    )

    best_model = ensemble["best_model"]

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    print("\nRunning validation ...")

    # Build per-context labels dict (single context, but validate_latent_circuit
    # accepts it for the trial-averaged R2 computation)
    ctx_labels = {
        "modality_context": mc_labels[mask],
    }

    val = validate_latent_circuit(
        best_model,
        u_ctx,
        z_ctx,
        y_ctx,
        w_rec,
        labels=ctx_labels,
        invariant_threshold=0.85,
        device=args.device,
    )

    corr: float = float(val["invariant_subspace_corr"])
    nmse_y: float = float(val["nmse_y"])
    inv_pass: bool = bool(val["invariant_subspace_pass"])

    # ------------------------------------------------------------------
    # Save best model
    # ------------------------------------------------------------------
    model_path = output_dir / "best_latent_circuit.pt"
    torch.save(best_model.state_dict(), str(model_path))
    print(f"Saved best model state_dict → {model_path}")

    # ------------------------------------------------------------------
    # Save validation_results.json (all values cast to Python builtins)
    # ------------------------------------------------------------------
    validation_results: dict = {
        "phase": "03-latent-circuit-inference",
        "plan": "06",
        "context": int(ctx),
        "n_latent": int(args.n_latent),
        "n_inits": int(args.n_inits),
        "epochs": int(args.epochs),
        "n_trials_used": n_trials_used,
        "device": str(args.device),
        "best_init_idx": int(ensemble["best_init_idx"]),
        "best_nmse_y": float(ensemble["best_nmse_y"]),
        "best_mse_z": float(ensemble["best_mse_z"]),
        "invariant_subspace_corr": corr,
        "invariant_subspace_pass": inv_pass,
        "invariant_subspace_threshold": 0.85,
        "activity_r2_full_space": float(val["activity_r2_full_space"]),
        "activity_r2_latent_space": float(val["activity_r2_latent_space"]),
        "trial_avg_r2_full_space": (
            float(val["trial_avg_r2_full_space"])
            if val["trial_avg_r2_full_space"] is not None
            else None
        ),
        "trial_avg_r2_latent_space": (
            float(val["trial_avg_r2_latent_space"])
            if val["trial_avg_r2_latent_space"] is not None
            else None
        ),
        "nmse_y": float(val["nmse_y"]),
        "nmse_q": float(val["nmse_q"]),
        "mse_z": float(val["mse_z"]),
        "status": "pass" if inv_pass else "soft-fail",
        "eval_procedure": "cluster_same_seed_as_train",
        "sigma_rec": 0.15,
    }
    val_path = output_dir / "validation_results.json"
    with open(val_path, "w") as fh:
        json.dump(validation_results, fh, indent=2)
    print(f"Saved validation results → {val_path}")

    # ------------------------------------------------------------------
    # Save ensemble_diagnostics.json
    # ------------------------------------------------------------------
    diagnostics: dict = {
        "all_nmse_y": [float(v) for v in ensemble["all_nmse_y"]],
        "all_mse_z": [float(v) for v in ensemble["all_mse_z"]],
        "convergence_stats": {
            "mean_nmse_y": float(np.mean(ensemble["all_nmse_y"])),
            "std_nmse_y": float(np.std(ensemble["all_nmse_y"])),
            "min_nmse_y": float(np.min(ensemble["all_nmse_y"])),
            "max_nmse_y": float(np.max(ensemble["all_nmse_y"])),
        },
    }
    diag_path = output_dir / "ensemble_diagnostics.json"
    with open(diag_path, "w") as fh:
        json.dump(diagnostics, fh, indent=2)
    print(f"Saved ensemble diagnostics → {diag_path}")

    # ------------------------------------------------------------------
    # Final summary line for cluster log scraping
    # ------------------------------------------------------------------
    print(
        f"\nSUMMARY n_trials_used={n_trials_used}, "
        f"corr={corr:.4f}, nmse_y={nmse_y:.4f}, "
        f"status={'PASS' if inv_pass else 'SOFT-FAIL'}"
    )


if __name__ == "__main__":
    main()
