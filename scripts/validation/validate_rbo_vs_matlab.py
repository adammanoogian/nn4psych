"""Cross-validate NumPyro compute_rbo_forward against MATLAB frugFun5.m port.

Generates synthetic helicopter task trials (CP and OB conditions), runs both
reference implementations with matched parameters, and compares the resulting
trajectories. Fails (exit 1) if any per-trial divergence exceeds tolerance.

Usage
-----
python scripts/validation/validate_rbo_vs_matlab.py [--alpha_tol FLOAT]
    [--omega_tol FLOAT] [--n_trials INT] [--output PATH]

The self-bucketing comparison strategy:
    1. Run MATLAB frugFun5 on synthetic bag positions to get the belief
       trajectory B[] (which IS the model's bucket position).
    2. Compute pred_errors = bag - B[:-1]  (MATLAB bucket trajectory).
    3. Feed those pred_errors to compute_rbo_forward so both models operate
       on the same state trajectory.
    This alignment is necessary because compute_rbo_forward takes pred_errors
    as input while frugFun5 takes raw bag positions and internally integrates
    the bucket position.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root and src/ so both config and nn4psych are importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import jax.numpy as jnp

from nn4psych.bayesian.reduced_bayesian import compute_rbo_forward, SIGMA_N
from nn4psych.bayesian._frugfun_reference import (
    frugfun5_reference,
    frugfun5_oddball_reference,
)


def generate_synthetic_trials(
    n_trials: int = 200,
    hazard_rate: float = 0.125,
    sigma_N: float = 20.0,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (helicopter_pos, bag_pos) synthetic trial arrays.

    Parameters
    ----------
    n_trials : int
        Number of trials to generate.
    hazard_rate : float
        Probability of a changepoint per trial.
    sigma_N : float
        Observation noise SD for bag placement.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    helicopter_pos : np.ndarray, shape (n_trials,)
    bag_pos : np.ndarray, shape (n_trials,), clipped to [0, 300]
    """
    rng = np.random.default_rng(rng_seed)
    helicopter_pos = np.zeros(n_trials)
    bag_pos = np.zeros(n_trials)
    helicopter_pos[0] = rng.uniform(80, 220)
    for i in range(n_trials):
        if i > 0 and rng.random() < hazard_rate:
            helicopter_pos[i] = rng.uniform(0, 300)
        elif i > 0:
            helicopter_pos[i] = helicopter_pos[i - 1]
        bag_pos[i] = np.clip(rng.normal(helicopter_pos[i], sigma_N), 0, 300)
    return helicopter_pos, bag_pos


def run_numpyro_cp(
    pred_errors: np.ndarray,
    H: float,
    LW: float,
    UU: float,
) -> dict[str, np.ndarray]:
    """Run compute_rbo_forward (changepoint context).

    Parameters
    ----------
    pred_errors : np.ndarray
        Prediction errors = bag - bucket (MATLAB belief trajectory).
    H, LW, UU : float
        Model parameters.

    Returns
    -------
    dict with keys 'alpha', 'omega', 'tau'
    """
    params = {
        "H": jnp.asarray(H),
        "LW": jnp.asarray(LW),
        "UU": jnp.asarray(UU),
    }
    lr, upd, omega, tau = compute_rbo_forward(
        params, jnp.asarray(pred_errors), "changepoint"
    )
    return {
        "alpha": np.asarray(lr),
        "omega": np.asarray(omega),
        "tau": np.asarray(tau),
    }


def run_numpyro_ob(
    pred_errors: np.ndarray,
    H: float,
    LW: float,
    UU: float,
) -> dict[str, np.ndarray]:
    """Run compute_rbo_forward (oddball context)."""
    params = {
        "H": jnp.asarray(H),
        "LW": jnp.asarray(LW),
        "UU": jnp.asarray(UU),
    }
    lr, upd, omega, tau = compute_rbo_forward(
        params, jnp.asarray(pred_errors), "oddball"
    )
    return {
        "alpha": np.asarray(lr),
        "omega": np.asarray(omega),
        "tau": np.asarray(tau),
    }


def compare_trajectories(
    numpyro_out: dict[str, np.ndarray],
    matlab_out: dict[str, np.ndarray],
    label: str,
) -> dict[str, float]:
    """Compute max-abs and median-abs diffs between trajectories.

    Parameters
    ----------
    numpyro_out : dict
        Output from run_numpyro_cp/ob with keys 'alpha', 'omega'.
    matlab_out : dict
        Output from frugfun5_reference with keys 'alpha', 'pCha'.
    label : str
        Scenario label for display.

    Returns
    -------
    dict with diff statistics.
    """
    alpha_diff = np.abs(numpyro_out["alpha"] - matlab_out["alpha"])
    omega_diff = np.abs(numpyro_out["omega"] - matlab_out["pCha"])

    diffs = {
        "alpha_max_abs": float(np.max(alpha_diff)),
        "alpha_median_abs": float(np.median(alpha_diff)),
        "omega_max_abs": float(np.max(omega_diff)),
        "omega_median_abs": float(np.median(omega_diff)),
    }

    print(f"\n=== {label} ===")
    for k, v in diffs.items():
        print(f"  {k}: {v:.6e}")

    # Print the first 5 per-trial values for diagnostics
    print(f"  First 5 MATLAB alpha: {matlab_out['alpha'][:5].round(4)}")
    print(f"  First 5 NumPyro alpha: {numpyro_out['alpha'][:5].round(4)}")
    print(f"  First 5 MATLAB pCha:   {matlab_out['pCha'][:5].round(4)}")
    print(f"  First 5 NumPyro omega: {numpyro_out['omega'][:5].round(4)}")

    return diffs


def main() -> int:
    """Run the three validation scenarios and report parity.

    Returns
    -------
    int
        0 if all scenarios pass tolerance gates, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="NumPyro vs MATLAB frugFun5 parity check"
    )
    parser.add_argument("--alpha_tol", type=float, default=1e-3)
    parser.add_argument("--omega_tol", type=float, default=1e-3)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/bayesian/matlab_parity_diffs.csv"),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NumPyro vs MATLAB frugFun5 parity check")
    print(f"n_trials={args.n_trials}, alpha_tol={args.alpha_tol}, "
          f"omega_tol={args.omega_tol}")
    print("=" * 60)

    _, bag = generate_synthetic_trials(n_trials=args.n_trials)

    rows = []
    scenarios = [
        ("changepoint_default", "cp", 0.125, 1.0, 1.0),
        ("changepoint_LW0.5", "cp", 0.125, 0.5, 1.0),
        ("oddball_default", "ob", 0.125, 1.0, 1.0),
    ]

    for scenario_label, context_type, H, LW, UU in scenarios:
        print(f"\n--- Scenario: {scenario_label} (H={H}, LW={LW}, UU={UU}) ---")

        # Step 1: Run MATLAB reference to get belief trajectory
        if context_type == "cp":
            matlab_out = frugfun5_reference(bag, Hazard=H, noise=SIGMA_N, likeWeight=LW)
        else:
            # OB uses the oddball variant (frugFun5_uniformOddballs.m)
            matlab_out = frugfun5_oddball_reference(
                bag, Hazard=H, noise=SIGMA_N, likeWeight=LW
            )

        # Step 2: Derive bucket positions from MATLAB belief trajectory
        # B has shape (n+1,); B[0..n-1] are the bucket positions at each trial
        bucket_positions = matlab_out["B"][:-1]  # shape (n,)
        pred_errors = bag - bucket_positions      # shape (n,)

        # Step 3: Run NumPyro on the same pred_errors
        if context_type == "cp":
            numpyro_out = run_numpyro_cp(pred_errors, H=H, LW=LW, UU=UU)
        else:
            numpyro_out = run_numpyro_ob(pred_errors, H=H, LW=LW, UU=UU)

        # Step 4: Compare
        diffs = compare_trajectories(numpyro_out, matlab_out, scenario_label)
        diffs["scenario"] = scenario_label
        diffs["H"] = H
        diffs["LW"] = LW
        diffs["UU"] = UU
        rows.append(diffs)

    # Write CSV
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote diff CSV to: {args.output}")

    # Gate decision
    fails = []
    for _, row in df.iterrows():
        if row["alpha_max_abs"] > args.alpha_tol:
            fails.append(
                f"{row['scenario']}: alpha_max_abs {row['alpha_max_abs']:.4e}"
                f" > tol {args.alpha_tol}"
            )
        if row["omega_max_abs"] > args.omega_tol:
            fails.append(
                f"{row['scenario']}: omega_max_abs {row['omega_max_abs']:.4e}"
                f" > tol {args.omega_tol}"
            )

    print("\n" + "=" * 60)
    if fails:
        print("FAIL: parity violated")
        for f in fails:
            print(f"  x {f}")
        return 1
    print("PASS: NumPyro and MATLAB agree within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
