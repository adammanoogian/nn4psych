#!/usr/bin/env python3
"""08: Latent Circuit Inference Pipeline (Phase 3 end-to-end).

Orchestrates the full Phase 3 workflow:
  1. Collect (u, z, y, task_active_mask) circuit data from a trained
     ContinuousActorCritic on ContextDecisionMaking-v0 [Plan 03-01].
  2. Fit a LatentNet ensemble at Wave A's chosen rank [Plan 03-02 + 03-03].
     Pass --masked to use masked-loss fitting (Gap 1, Plan 03-05).
  3. Validate (invariant subspace, activity R^2, trial-avg R^2) [Plan 03-02].
  4. Perturb the RNN via Q-mapped rank-one perturbations [Plan 03-04].

The chosen rank is read from
output/circuit_analysis/n_latent_sweep/wave_a_selection.json (produced by
Plan 03-03 Wave A). When --skip_fitting is passed, this script loads
output/circuit_analysis/best_latent_circuit_waveA.pt (Wave A's chosen Q),
NOT the legacy 03-02 benchmark output/circuit_analysis/best_latent_circuit.pt.

For masked-loss fitting (--masked): the script reads task_active_mask from
circuit_data.npz and passes it to fit_latent_circuit_ensemble. Artifacts land
under output/circuit_analysis/n_latent_sweep_masked/n{N}/ so they never
overwrite Wave A's output/circuit_analysis/n_latent_sweep/ artifacts.

Preconditions
-------------
- data/processed/rnn_behav/model_context_dm_dual.pth MUST be the
  ContinuousActorCritic-format checkpoint produced by
  cluster/run_circuit_ensemble.sh's embedded retrain (03-02). The earlier
  03-01-era plain ActorCritic checkpoint is incompatible — load_state_dict
  will raise a key-mismatch error if the older file is in place. If running
  locally and you have not pulled the cluster's updated checkpoint, fetch it
  before invoking this script.

Usage
-----
    # Full pipeline (~hours; rarely needed locally — use cluster for fitting)
    python scripts/data_pipeline/08_infer_latent_circuits.py

    # Quick smoke test (5 inits, 50 epochs, 50 trials per context, 20 eval trials)
    python scripts/data_pipeline/08_infer_latent_circuits.py --quick

    # Quick smoke test with masked-loss fitting (Gap 1)
    python scripts/data_pipeline/08_infer_latent_circuits.py --quick --masked

    # Use existing data + Wave A's fitted Q; only run perturbation step
    python scripts/data_pipeline/08_infer_latent_circuits.py --skip_collection --skip_fitting
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path so nn4psych and envs are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nn4psych.models.continuous_rnn import ContinuousActorCritic  # noqa: E402
from nn4psych.analysis.circuit_inference import (  # noqa: E402
    collect_circuit_data,
    save_circuit_data,
    fit_latent_circuit_ensemble,
    validate_latent_circuit,
    perturb_and_evaluate,
)
from nn4psych.analysis.latent_net import LatentNet  # noqa: E402
from nn4psych.training.resources import configure_cpu_threads  # noqa: E402
from envs.neurogym_wrapper import SingleContextDecisionMakingWrapper  # noqa: E402


WAVE_A_SELECTION_PATH = Path("output/circuit_analysis/n_latent_sweep/wave_a_selection.json")
WAVE_A_BEST_PT_PATH = Path("output/circuit_analysis/best_latent_circuit_waveA.pt")
# Masked-sweep (03-05 Gap 1) artifacts live under n_latent_sweep_masked/
WAVE_A_MASKED_SELECTION_PATH = Path(
    "output/circuit_analysis/n_latent_sweep_masked/wave_a_masked_selection.json"
)


def load_wave_a_chosen_rank(path: Path = WAVE_A_SELECTION_PATH) -> int:
    """Read Wave A's chosen rank from the selection JSON.

    Parameters
    ----------
    path : Path, optional
        Default output/circuit_analysis/n_latent_sweep/wave_a_selection.json.

    Returns
    -------
    int
        The chosen n_latent value.

    Raises
    ------
    FileNotFoundError
        If wave_a_selection.json does not exist (Wave A / Plan 03-03 not run).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Wave A selection file not found: {path}. "
            "Run Plan 03-03 (Wave A n_latent sweep) before 03-04. "
            "Expected location: output/circuit_analysis/n_latent_sweep/"
            "wave_a_selection.json"
        )
    sel = json.loads(path.read_text())
    return int(sel["chosen_rank"])


def load_wave_a_masked_chosen_rank(
    path: Path = WAVE_A_MASKED_SELECTION_PATH,
) -> int | None:
    """Read the masked Wave A chosen rank from the selection JSON.

    Returns None (fall-through) if the file does not yet exist, so callers can
    gracefully handle the case where the masked sweep has not completed.

    Parameters
    ----------
    path : Path, optional
        Default output/circuit_analysis/n_latent_sweep_masked/
        wave_a_masked_selection.json.

    Returns
    -------
    int or None
        The chosen n_latent from the masked sweep, or None if unavailable.
    """
    if not path.exists():
        return None
    sel = json.loads(path.read_text())
    return int(sel["chosen_rank"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/processed/rnn_behav/model_context_dm_dual.pth",
    )
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--input_dim", type=int, default=7)
    parser.add_argument("--action_dim", type=int, default=3)
    # Data
    parser.add_argument("--n_trials_per_context", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=75)
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/rnn_behav/circuit_data.npz",
    )
    # Fitting (n_latent comes from Wave A, NOT a CLI arg)
    parser.add_argument("--n_inits", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l_y", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    # Perturbation
    parser.add_argument("--n_eval_trials", type=int, default=200)
    parser.add_argument("--n_baseline_runs", type=int, default=5)
    parser.add_argument("--n_top_connections", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    # Validation
    parser.add_argument("--invariant_threshold", type=float, default=0.85)
    # Output
    parser.add_argument("--output_dir", type=str, default="output/circuit_analysis")
    # Flags
    parser.add_argument(
        "--quick",
        action="store_true",
        help="5 inits, 50 epochs, 50 trials per context, 20 eval trials",
    )
    parser.add_argument("--skip_collection", action="store_true")
    parser.add_argument(
        "--skip_fitting",
        action="store_true",
        help="Load best_latent_circuit_waveA.pt (Wave A's chosen Q); skip fitting",
    )
    parser.add_argument(
        "--masked",
        action="store_true",
        help=(
            "Use masked-loss fitting (Gap 1, Plan 03-05): compute NMSE_y and mse_z "
            "over task-active timesteps only (stimulus+decision periods). "
            "Requires circuit_data.npz to contain 'task_active_mask'. "
            "Artifacts land under n_latent_sweep_masked/ not n_latent_sweep/."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run the end-to-end Phase 3 latent circuit inference pipeline.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    configure_cpu_threads()
    args = parse_args()

    # --masked: redirect default output dir to n_latent_sweep_masked/ BEFORE
    # --quick path override so both interact correctly
    if args.masked and args.output_dir == "output/circuit_analysis":
        # Will be further qualified by n_latent after we read Wave A rank;
        # we set the root here and append /n{N}/ below
        args.output_dir = "output/circuit_analysis/n_latent_sweep_masked"

    if args.quick:
        args.n_inits = 5
        args.epochs = 50
        args.n_trials_per_context = 50
        args.n_eval_trials = 20
        args.n_baseline_runs = 2
        # Redirect data and output paths so --quick never overwrites canonical
        # artifacts. The smoke run's circuit_data goes to a separate subdir.
        # This applies regardless of --masked (both paths write to smoke_test/).
        if not args.skip_collection:
            args.data_path = str(
                Path(args.data_path).parent / "smoke_test" / Path(args.data_path).name
            )
        if not args.skip_fitting:
            args.output_dir = str(Path(args.output_dir).parent / "smoke_test")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Wave A: chosen rank (locked — never hardcode) ----
    n_latent = load_wave_a_chosen_rank()
    print(f"Wave A chosen rank: n_latent={n_latent}")

    # ---- Step 1: Load model ----
    model = ContinuousActorCritic(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        alpha=0.2,
        sigma_rec=0.15,
        gain=0.9,
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    print(f"Loaded model: {args.model_path}")

    # ---- Step 2: Collect (or load) circuit data ----
    if not args.skip_collection:
        data = collect_circuit_data(
            model,
            SingleContextDecisionMakingWrapper,
            modality_contexts=[0, 1],
            n_trials_per_context=args.n_trials_per_context,
            max_steps=args.max_steps,
        )
        save_circuit_data(data, str(Path(args.data_path).parent))

    data_npz = np.load(args.data_path)
    u, z, y = data_npz["u"], data_npz["z"], data_npz["y"]
    labels = {
        "modality_context": data_npz["labels_modality_context"],
        "coherence_sign": data_npz["labels_coherence_sign"],
        "correct_action": data_npz["labels_correct_action"],
    }
    print(f"Circuit data: u={u.shape}, z={z.shape}, y={y.shape}")

    # Load task_active_mask when --masked is requested
    task_active_mask = None
    if args.masked:
        if "task_active_mask" not in data_npz.files:
            raise KeyError(
                f"--masked requested but circuit_data.npz at '{args.data_path}' "
                "lacks 'task_active_mask' key. "
                "Regenerate circuit_data.npz via Task 1D of Plan 03-05 "
                "(run collect_circuit_data with the updated circuit_inference.py)."
            )
        task_active_mask = data_npz["task_active_mask"]  # (n_trials, T), bool
        active_per_trial = task_active_mask.sum(axis=1)
        print(
            f"task_active_mask: mean_active={float(active_per_trial.mean()):.1f}, "
            f"min={int(active_per_trial.min())}, max={int(active_per_trial.max())} "
            f"steps (of T={u.shape[1]})"
        )

    # ---- Step 3: Fit (or load Wave A's chosen Q) ----
    if not args.skip_fitting:
        result = fit_latent_circuit_ensemble(
            u,
            z,
            y,
            n_inits=args.n_inits,
            n_latent=n_latent,
            epochs=args.epochs,
            lr=args.lr,
            l_y=args.l_y,
            weight_decay=args.weight_decay,
            task_active_mask=task_active_mask,  # None → unmasked; set → Gap-1 masked
        )
        # Choose artifact filename based on masked vs standard path
        if args.masked:
            pt_name = "best_latent_circuit.pt"
            diag_name = "ensemble_diagnostics.json"
        else:
            # Preserve Wave A archive naming convention
            pt_name = "best_latent_circuit_waveB_refit.pt"
            diag_name = "ensemble_diagnostics_waveB_refit.json"

        torch.save(result["best_model"].state_dict(), output_dir / pt_name)
        with open(output_dir / diag_name, "w") as f:
            json.dump(
                {
                    "all_nmse_y": [float(v) for v in result["all_nmse_y"]],
                    "all_nmse_y_full": [float(v) for v in result["all_nmse_y_full"]],
                    "all_mse_z": [float(v) for v in result["all_mse_z"]],
                    "best_init_idx": int(result["best_init_idx"]),
                    "best_nmse_y": float(result["best_nmse_y"]),
                    "best_nmse_y_full": float(result["best_nmse_y_full"]),
                    "masked": bool(result["masked"]),
                    "n_latent": int(result["n_latent"]),
                    "n_inits": int(result["n_inits"]),
                },
                f,
                indent=2,
            )
        best_model = result["best_model"]
    else:
        if not WAVE_A_BEST_PT_PATH.exists():
            raise FileNotFoundError(
                f"--skip_fitting requires Wave A's saved Q at {WAVE_A_BEST_PT_PATH}. "
                "Run Plan 03-03 (Wave A) first."
            )
        best_model = LatentNet(
            n=n_latent,
            N=args.hidden_dim,
            input_size=u.shape[2],
            n_trials=u.shape[0],
            output_size=z.shape[2],
        )
        best_model.load_state_dict(
            torch.load(WAVE_A_BEST_PT_PATH, map_location="cpu")
        )
        # Recompute Q via Cayley transform (q is not a parameter; it is derived from a)
        best_model.q = best_model.cayley_transform(best_model.a)

        # Assert Q orthonormality: ||Q Q^T - I|| < 1e-4
        q_arr = best_model.q.detach()
        qqt = q_arr @ q_arr.T
        identity_err = float(torch.norm(qqt - torch.eye(q_arr.shape[0])).item())
        assert identity_err < 1e-4, (
            f"--skip_fitting: Q is not orthonormal after cayley_transform "
            f"(||Q Q^T - I|| = {identity_err:.6e}). State dict may be corrupted, "
            f"or wave_a_selection.json's chosen_rank ({n_latent}) does not match "
            f"the saved best_latent_circuit_waveA.pt."
        )
        print(f"--skip_fitting: Wave A Q loaded; orthonormality err={identity_err:.2e}")

    # ---- Step 4: Validate ----
    W_rec = model.W_hh.weight.data.detach().numpy()
    W_in = model.W_ih.weight.data.detach().numpy()
    val = validate_latent_circuit(
        best_model,
        u,
        z,
        y,
        W_rec,
        W_in,
        labels=labels,
        invariant_threshold=args.invariant_threshold,
    )

    def _to_python(v: object) -> object:
        """Convert numpy scalar to Python builtin for json.dump."""
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, dict):
            return {_to_python(k): _to_python(vv) for k, vv in v.items()}
        return v

    val_report = {
        k: _to_python(v) for k, v in val.items()
    }
    val_report["data_shape"] = {
        "n_trials": int(u.shape[0]),
        "T": int(u.shape[1]),
        "N": int(y.shape[2]),
    }
    val_report["wave_a_chosen_rank"] = int(n_latent)
    val_report["masked"] = bool(args.masked)  # downstream 03-08 can disambiguate

    # Choose output filename: masked runs write to validation_results.json
    # (same name cluster scripts expect); waveB refit writes with suffix
    val_out_name = "validation_results.json" if args.masked else "validation_results_waveB.json"
    with open(output_dir / val_out_name, "w") as f:
        json.dump(val_report, f, indent=2)
    print(f"Invariant subspace corr: {val['invariant_subspace_corr']:.4f}")
    if val['trial_avg_r2_full_space'] is not None:
        print(f"Trial-avg R2 (full):     {val['trial_avg_r2_full_space']:.4f}")
    print(f"Masked fit:              {bool(args.masked)}")

    # ---- Step 5: Perturb (skip for masked sweep — separate analysis) ----
    if not args.masked:
        pert = perturb_and_evaluate(
            model,
            best_model,
            SingleContextDecisionMakingWrapper,
            n_eval_trials=args.n_eval_trials,
            n_baseline_runs=args.n_baseline_runs,
            n_top_connections=args.n_top_connections,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        with open(output_dir / "perturbation_results.json", "w") as f:
            json.dump(pert, f, indent=2)

        print(f"\nPerturbations tested:        {len(pert['perturbations'])}")
        print(f"Baseline std (pooled):       {pert['baseline']['std_reward_pooled']:.4f}")
        print(
            f"Significance threshold:      "
            f"{pert['baseline']['significance_threshold']:.4f}"
        )
        print(f"Mean |reward delta|:         {pert['summary']['mean_abs_reward_delta']:.4f}")
        print(f"Significant perturbations:   {pert['summary']['n_significant']}")
    else:
        print("\n(--masked: perturbation step skipped — separate analysis in 03-04/03-08)")

    print(f"\nAll results in {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
