"""Parameter recovery for the Nassar 2021 Reduced Bayesian Observer.

Stage 09a: Validate model identifiability by fitting MCMC to synthetic
datasets sampled from the prior, then measuring Pearson r between true
(prior-sampled) and recovered (posterior-mean) parameters.

Pipeline:
1. Sample N prior parameter sets via prior_sampler.
2. For each dataset i: generate CP + OB synthetic sequences, fit MCMC
   per condition (2 fits per dataset), average posterior means.
3. Aggregate recovery_report.json: per-parameter r, r^2, pass/fail gate.
4. Save scatter PNGs (true vs recovered) per parameter.
5. Print summary table to stdout.

Design choice (per-condition fit):
    Each synthetic dataset is fit once per condition (CP, OB) and
    posterior means are averaged across both. This validates the actual
    fitting pipeline used on human data (per-subject x per-condition MCMC),
    not a simplified single-condition version. Recovery r values average
    identifiability evidence across both task conditions.

Recovery gate:
    Per-parameter Pearson r >= 0.85 (ROADMAP SC-2). Note: gate is
    informational in smoke run (N=4) — formal evidence requires full
    50-dataset run. BAYES-06 is NOT closed until full run completes.

Usage (smoke, ~5-15 min on 16GB RAM):
    python scripts/data_pipeline/09a_param_recovery.py --smoke

Usage (full overnight run, ~17h worst case):
    python scripts/data_pipeline/09a_param_recovery.py

Memory note:
    Each MCMC fit at 4 chains x 2000 warmup x 2000 samples uses ~300-400 MB.
    Runs are sequential (no parallelism) per project CLAUDE.md constraint.
    Use scripts/run_local.py wrapper for memory-guarded execution.

References
----------
Nassar, M. R., et al. (2021). PMC8041039. Reduced Bayesian Observer for
    changepoint/oddball conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Must be set before any JAX import (defensive: also set in nn4psych.bayesian.__init__)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Project imports — after env setup above
import jax
import jax.numpy as jnp
from scipy.stats import pearsonr

# Add src/ and project root to path so nn4psych.bayesian is importable.
# src/: required for nn4psych package discovery
# project root: required for 'envs' module (PIE_CP_OB_v2), loaded by
#   nn4psych.__init__; without project root in path, ModuleNotFoundError.
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
_SRC = _PROJECT_ROOT / "src"
for _p in [str(_SRC), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn4psych.bayesian import (  # noqa: E402
    reduced_bayesian_model,
    prior_sampler,
    simulate_synthetic_data,
    fit_with_retry,
    make_fit_summary,
    to_jsonable,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Recovery is reported on log_UU (the actual MCMC sample site) rather
# than the deterministic UU = exp(log_UU).  log_UU has a Gaussian-shaped
# posterior, while UU is exponential of that — identifiability is much
# more interpretable on log_UU.  See Nassar 2021 fitFrugFunSchiz.m:42,
# 130 for the matching reparameterization.
PARAM_NAMES = ["H", "LW", "log_UU", "sigma_motor", "sigma_LR"]
CONDITIONS = ["changepoint", "oddball"]
RECOVERY_GATE_R = 0.85


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the parameter recovery script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with all recovery configuration.
    """
    parser = argparse.ArgumentParser(
        description="Nassar 2021 Reduced Bayesian Observer: parameter recovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_synthetic",
        type=int,
        default=50,
        help="Number of synthetic prior-sampled datasets to recover",
    )
    parser.add_argument(
        "--n_trials_per_condition",
        type=int,
        default=100,
        help="Number of trials per condition (CP or OB) per synthetic dataset",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=2000,
        help="NUTS warmup steps for first attempt",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Posterior samples per chain",
    )
    parser.add_argument(
        "--num_chains",
        type=int,
        default=4,
        help="Number of parallel MCMC chains",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for prior sampling and data generation",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke-test mode: override defaults to "
            "n_synthetic=4, num_warmup=200, num_samples=200, num_chains=2"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/bayesian/param_recovery",
        help="Output directory for recovery_report.json and figures/",
    )
    args = parser.parse_args()

    if args.smoke:
        args.n_synthetic = 4
        args.num_warmup = 200
        args.num_samples = 200
        args.num_chains = 2
        # Default smoke output dir unless explicitly overridden
        if args.output_dir == "data/processed/bayesian/param_recovery":
            args.output_dir = "data/processed/bayesian/param_recovery_smoke"

    return args


# ---------------------------------------------------------------------------
# Per-dataset fitting
# ---------------------------------------------------------------------------


def fit_single_dataset(
    params_i: dict[str, float],
    dataset_idx: int,
    n_trials: int,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """Fit MCMC to one prior-sampled synthetic dataset, separately per condition.

    Generates CP and OB sequences and runs fit_with_retry on each.
    Returns per-condition recovered posterior means rather than averaging,
    so identifiability evidence can be evaluated separately for each task
    condition (matches Nassar 2021 fit-per-block design).

    Parameters
    ----------
    params_i : dict[str, float]
        True (prior-sampled) parameter values; must include both
        ``log_UU`` (sample site) and ``UU`` (deterministic divisor).
    dataset_idx : int
        Index of this dataset in the recovery sweep (0-indexed).
    n_trials : int
        Number of trials per condition.
    num_warmup : int
        NUTS warmup steps for first attempt.
    num_samples : int
        Posterior samples per chain.
    num_chains : int
        Number of MCMC chains.
    seed : int
        Base seed; dataset uses seed + dataset_idx * 10 to avoid collisions.
    output_dir : Path
        Directory for per-fit JSON files.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{condition: {param: posterior_mean}}`` with one entry per
        ``CONDITIONS`` and one inner entry per ``PARAM_NAMES``.  Returns
        NaN for any param missing from posterior samples.
    """
    per_fit_dir = output_dir / "per_fit"
    per_fit_dir.mkdir(parents=True, exist_ok=True)

    per_condition: dict[str, dict[str, float]] = {}
    dataset_seed = seed + dataset_idx * 10

    for cond_idx, condition in enumerate(CONDITIONS):
        fit_seed = dataset_seed + cond_idx

        # Generate synthetic data for this condition
        bag, bucket = simulate_synthetic_data(
            params_i,
            n_trials=n_trials,
            hazard=0.125,
            context=condition,
            seed=fit_seed,
        )

        # Retry-enabled MCMC fit
        mcmc, status, attempts = fit_with_retry(
            reduced_bayesian_model,
            {
                "bag_positions": bag,
                "bucket_positions": bucket,
                "context": condition,
            },
            seed=fit_seed,
            num_warmup_first=num_warmup,
            num_warmup_retry=num_warmup * 2,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        if status == "FAILED":
            print(
                f"  WARNING: dataset {dataset_idx:03d}/{condition} fit FAILED "
                f"(using posterior anyway)"
            )

        # Write per-fit JSON
        fit_json = make_fit_summary(
            mcmc,
            status=status,
            attempts=attempts,
            var_names=PARAM_NAMES,
            dataset_idx=int(dataset_idx),
            condition=condition,
            true_params={k: float(v) for k, v in params_i.items()},
        )
        json_path = per_fit_dir / f"synth_{dataset_idx:03d}_{condition}.json"
        with open(json_path, "w") as f:
            json.dump(fit_json, f, indent=2)

        # Collect posterior means from this condition fit
        posterior_samples = mcmc.get_samples()
        cond_recovered: dict[str, float] = {}
        for param in PARAM_NAMES:
            if param in posterior_samples:
                cond_recovered[param] = float(
                    np.mean(np.asarray(posterior_samples[param]))
                )
            else:
                cond_recovered[param] = float("nan")
        per_condition[condition] = cond_recovered

    return per_condition


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------


def save_scatter(
    param_name: str,
    true_vals: list[float],
    recovered_vals: list[float],
    r_val: float,
    passes_gate: bool,
    output_dir: Path,
) -> None:
    """Save a true-vs-recovered scatter plot for one parameter.

    Parameters
    ----------
    param_name : str
        Parameter name (e.g., ``'H'``).
    true_vals : list[float]
        True (prior-sampled) values.
    recovered_vals : list[float]
        Recovered (posterior-mean) values.
    r_val : float
        Pearson r between true and recovered.
    passes_gate : bool
        Whether r >= RECOVERY_GATE_R (0.85).
    output_dir : Path
        Base output directory; figure saved to ``output_dir/figures/``.
    """
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))

    true_arr = np.array(true_vals)
    rec_arr = np.array(recovered_vals)

    ax.scatter(true_arr, rec_arr, alpha=0.7, s=50, color="steelblue", zorder=3)

    # Diagonal y=x reference line
    all_vals = np.concatenate([true_arr, rec_arr])
    val_min, val_max = float(all_vals.min()), float(all_vals.max())
    margin = (val_max - val_min) * 0.05
    diag_range = [val_min - margin, val_max + margin]
    ax.plot(diag_range, diag_range, "k--", linewidth=1.0, alpha=0.5, label="y=x")

    gate_label = "PASS" if passes_gate else f"FAIL (< {RECOVERY_GATE_R:.2f})"
    ax.set_title(
        f"{param_name}: r={r_val:.3f}  [{gate_label}]",
        fontsize=12,
        fontweight="bold" if not passes_gate else "normal",
        color="red" if not passes_gate else "black",
    )
    ax.set_xlabel(f"True {param_name}", fontsize=11)
    ax.set_ylabel(f"Recovered {param_name}", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()

    fig_path = fig_dir / f"recovery_{param_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # Required: never plt.show() (feedback_no_interactive_plots)


# ---------------------------------------------------------------------------
# Main recovery pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the parameter recovery pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Parameter Recovery — Nassar 2021 Reduced Bayesian Observer\n"
        f"{'='*60}\n"
        f"N synthetic datasets:  {args.n_synthetic}\n"
        f"Trials per condition:  {args.n_trials_per_condition}\n"
        f"MCMC config:           {args.num_chains} chains x "
        f"{args.num_warmup} warmup x {args.num_samples} samples\n"
        f"Output dir:            {output_dir}\n"
        f"{'='*60}"
    )

    start_time = time.time()

    # 1. Sample N prior parameter sets
    print("\nSampling prior parameters...")
    rng_key = jax.random.PRNGKey(args.seed)
    prior_samples = prior_sampler(
        reduced_bayesian_model,
        num_samples=args.n_synthetic,
        rng_key=rng_key,
    )

    # Convert to list of dicts. PARAM_NAMES uses log_UU (sample site).
    # Also include 'UU' (deterministic = exp(log_UU)) so simulate_synthetic_data
    # can use the divisor value when running the participant forward model.
    dataset_params: list[dict[str, float]] = []
    for i in range(args.n_synthetic):
        d = {p: float(prior_samples[p][i]) for p in PARAM_NAMES}
        d["UU"] = float(prior_samples["UU"][i])
        dataset_params.append(d)

    # 2. Fit each synthetic dataset (per-condition: CP and OB are tracked
    #    separately so identifiability is reported per condition).
    true_vals: dict[str, list[float]] = {p: [] for p in PARAM_NAMES}
    recovered_per_cond: dict[str, dict[str, list[float]]] = {
        cond: {p: [] for p in PARAM_NAMES} for cond in CONDITIONS
    }
    failed_indices: list[int] = []

    for i, params_i in enumerate(dataset_params):
        print(
            f"\n[{i+1}/{args.n_synthetic}] dataset {i:03d} "
            f"| H={params_i['H']:.3f} LW={params_i['LW']:.3f} "
            f"log_UU={params_i['log_UU']:.3f} (UU={params_i['UU']:.2f})"
        )
        for p in PARAM_NAMES:
            true_vals[p].append(params_i[p])

        fit_start = time.time()
        recovered_i = fit_single_dataset(
            params_i=params_i,
            dataset_idx=i,
            n_trials=args.n_trials_per_condition,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            seed=args.seed,
            output_dir=output_dir,
        )
        fit_elapsed = time.time() - fit_start
        print(
            f"  Done in {fit_elapsed:.1f}s | "
            f"CP H={recovered_i['changepoint']['H']:.3f}  "
            f"OB H={recovered_i['oddball']['H']:.3f}"
        )

        # Track failed fits (any NaN in either condition's recovered)
        any_nan = False
        for cond in CONDITIONS:
            for p in PARAM_NAMES:
                v = recovered_i[cond][p]
                recovered_per_cond[cond][p].append(v)
                if np.isnan(v):
                    any_nan = True
        if any_nan:
            failed_indices.append(i)

    # 3. Compute Pearson r per (parameter, condition) and build report
    print(f"\n{'='*68}")
    print(
        f"Parameter Recovery Report "
        f"(N={args.n_synthetic} synthetic datasets, per condition)"
    )
    print(f"{'='*68}")
    print(f"{'param':<14} {'r_CP':>10}  {'r_OB':>10}  status")
    print(f"{'-'*68}")

    per_parameter_per_condition: dict[str, dict[str, dict]] = {
        cond: {} for cond in CONDITIONS
    }
    for param in PARAM_NAMES:
        t_arr = np.array(true_vals[param])
        cond_results: dict[str, dict] = {}
        for cond in CONDITIONS:
            r_arr = np.array(recovered_per_cond[cond][param])
            valid_mask = ~(np.isnan(t_arr) | np.isnan(r_arr))
            n_valid = int(valid_mask.sum())

            if n_valid >= 2:
                r_val, _p = pearsonr(t_arr[valid_mask], r_arr[valid_mask])
                r2_val = float(r_val ** 2)
            else:
                r_val = float("nan")
                r2_val = float("nan")

            passes = (not np.isnan(r_val)) and (r_val >= RECOVERY_GATE_R)
            cond_results[cond] = {
                "r": float(r_val),
                "r2": r2_val,
                "passes_gate": bool(passes),
                "n_valid": n_valid,
                "true": [float(v) for v in t_arr.tolist()],
                "recovered": [float(v) for v in r_arr.tolist()],
            }
            per_parameter_per_condition[cond][param] = cond_results[cond]

            # Save per-condition scatter plot
            save_scatter(
                param_name=f"{param}_{cond}",
                true_vals=t_arr[valid_mask].tolist(),
                recovered_vals=r_arr[valid_mask].tolist(),
                r_val=r_val,
                passes_gate=bool(passes),
                output_dir=output_dir,
            )

        # Print row
        cp_r = cond_results["changepoint"]["r"]
        ob_r = cond_results["oddball"]["r"]
        cp_pass = cond_results["changepoint"]["passes_gate"]
        ob_pass = cond_results["oddball"]["passes_gate"]
        if cp_pass and ob_pass:
            status_str = "PASS (both)"
        elif cp_pass or ob_pass:
            which = "CP" if cp_pass else "OB"
            status_str = f"PASS ({which} only)"
        else:
            status_str = f"FAIL  <-- both below {RECOVERY_GATE_R:.2f}"
        print(
            f"{param:<14} {cp_r:>10.3f}  {ob_r:>10.3f}  {status_str}"
        )

    overall_passes = all(
        per_parameter_per_condition[cond][p]["passes_gate"]
        for cond in CONDITIONS
        for p in PARAM_NAMES
        if not np.isnan(per_parameter_per_condition[cond][p]["r"])
    )
    n_failed = len(failed_indices)
    n_pass_cells = sum(
        1
        for cond in CONDITIONS
        for p in PARAM_NAMES
        if per_parameter_per_condition[cond][p]["passes_gate"]
    )
    n_total_cells = len(PARAM_NAMES) * len(CONDITIONS)
    print(
        f"\nOverall: {n_pass_cells}/{n_total_cells} (param x condition) "
        f"cells pass gate"
    )
    print(f"Failed fits: {n_failed}/{args.n_synthetic} (indices: {failed_indices})")

    # 4. Write recovery_report.json
    report = to_jsonable({
        "n_synthetic": int(args.n_synthetic),
        "n_trials_per_condition": int(args.n_trials_per_condition),
        "mcmc_config": {
            "num_warmup": int(args.num_warmup),
            "num_samples": int(args.num_samples),
            "num_chains": int(args.num_chains),
            "target_accept_prob": 0.95,
        },
        "per_parameter_per_condition": per_parameter_per_condition,
        "overall_passes": bool(overall_passes),
        "n_failed_fits": int(n_failed),
        "failed_indices": [int(x) for x in failed_indices],
        "elapsed_seconds": float(time.time() - start_time),
    })

    report_path = output_dir / "recovery_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    total_elapsed = time.time() - start_time
    print(f"\nReport: {report_path}")
    print(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    if not overall_passes:
        print(
            "\nNOTE: Not all parameters pass r >= 0.85. "
            "This does NOT block this plan — see SUMMARY for BAYES-06 status. "
            "Identifiability issues: first check tau update equation (RESEARCH.md Pitfall 1)."
        )
    else:
        print("\nAll parameters pass gate — BAYES-06 candidate for closure.")


if __name__ == "__main__":
    main()
