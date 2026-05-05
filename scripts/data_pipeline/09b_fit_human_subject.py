"""Fit Reduced Bayesian Observer to one Nassar 2021 subject × condition.

Stage 09b: Per-subject MCMC fit on real human behavioral data from the
Brain 2021 (Nassar/Waltz/Albrecht/Gold/Frank) AASP task.  Designed to be
called once per (subject_idx, condition) pair so SLURM array jobs can
parallelize the 134-subject × 2-condition sweep (268 fits total).

Pipeline:
1. Load ``data/processed/nassar2021/subject_trials.npy`` (134 dicts).
2. Filter to one subject, combine avoid+seek into the requested noise
   structure (CP = condition codes 0,1; OB = condition codes 2,3) —
   matches Nassar 2021 ``AASP_mastList.m:133`` which pools across valence
   for cognitive parameter fits.
3. Compute newBlock mask at avoid→seek boundary so ``compute_rbo_forward``
   resets relative uncertainty (matches ``getTrialVarsFromPEs_cannon.m:117``).
4. Run NUTS MCMC with retry-on-failure.
5. Compute posterior predictive bucket-update means and behavioral metrics
   (Pearson r, RMSE, MAE between observed and predicted updates).
6. Save per-fit JSON (summary) + NPZ (raw posterior samples).

Usage (single fit, local smoke test):
    python scripts/data_pipeline/09b_fit_human_subject.py \\
        --subject_idx 0 --condition changepoint \\
        --num_warmup 1000 --num_samples 1000 --num_chains 2

Usage (full BAYES-06 grade, called by SLURM array):
    python scripts/data_pipeline/09b_fit_human_subject.py \\
        --subject_idx 42 --condition oddball

Output:
    {output_dir}/per_fit/{subject_id}_{condition}.json   — summary
    {output_dir}/per_fit/{subject_id}_{condition}.npz    — raw samples

References
----------
Nassar, M. R., Waltz, J. A., Albrecht, M. A., Gold, J. M., & Frank, M. J.
    (2021). All or nothing belief updating in patients with schizophrenia
    reduces precision and flexibility of beliefs.  Brain, 144(3), 1013-1029.
    https://doi.org/10.1093/brain/awaa453
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Must be set before any JAX import
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import pearsonr

# Add src/ and project root to path
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
_SRC = _PROJECT_ROOT / "src"
for _p in [str(_SRC), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn4psych.bayesian import (  # noqa: E402
    reduced_bayesian_model,
    fit_with_retry,
    make_fit_summary,
    to_jsonable,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARAM_NAMES = ["H", "LW", "log_UU", "sigma_motor", "sigma_LR"]
DETERMINISTIC_NAMES = ["learning_rate", "normative_update", "omega", "tau"]

# AASP condition coding from extract_nassar_trials.py:
#   0 = cloud_cp_avoid     | 1 = cloud_cp_seek      → changepoint (CP)
#   2 = cloud_drift_avoid  | 3 = cloud_drift_seek   → oddball (OB / drift)
# Per Nassar 2021 AASP_mastList.m:133, valence (avoid/seek) is pooled
# for cognitive parameter fits.
CONDITION_CODES = {
    "changepoint": [0, 1],
    "oddball": [2, 3],
}

DEFAULT_DATA_PATH = "data/processed/nassar2021/subject_trials.npy"
DEFAULT_OUTPUT_DIR = "data/processed/bayesian/human_validation"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the per-subject fitter.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fit Nassar 2021 Reduced Bayesian Observer to one subject × condition"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subject_idx",
        type=int,
        required=True,
        help="0-based subject index into subject_trials.npy (0..133)",
    )
    parser.add_argument(
        "--condition",
        choices=["changepoint", "oddball"],
        required=True,
        help="Noise structure to fit (avoid+seek pooled within structure)",
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
        help="Base random seed (fit seed = base + subject_idx*100 + cond_idx)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to subject_trials.npy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for per_fit/{subject}_{condition}.json + .npz",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_subject_data(
    data_path: Path, subject_idx: int, condition: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load and filter one subject's data for one noise structure.

    Combines avoid+seek blocks within the requested noise structure
    (changepoint or oddball/drift), per Nassar 2021 fit convention.

    Parameters
    ----------
    data_path : Path
        Path to ``subject_trials.npy``.
    subject_idx : int
        0-based subject index.
    condition : str
        ``"changepoint"`` or ``"oddball"``.

    Returns
    -------
    bag_positions : np.ndarray
        Outcome (bag landing) positions for the selected trials.
    bucket_positions : np.ndarray
        Prediction (bucket) positions for the selected trials.
    new_block_mask : np.ndarray
        Boolean array marking the first trial of each block.  Set True at
        the avoid→seek boundary within the noise structure (and at trial 0).
    metadata : dict
        ``{'subject_id', 'is_patient', 'n_trials', 'n_blocks', 'sub_blocks'}``.

    Raises
    ------
    ValueError
        If subject_idx is out of range or no trials are found for the
        requested condition.
    """
    data = np.load(str(data_path), allow_pickle=True)
    if subject_idx < 0 or subject_idx >= len(data):
        raise ValueError(
            f"subject_idx out of range: expected 0..{len(data)-1}, got {subject_idx}"
        )

    subject = data[subject_idx]
    cond_codes = CONDITION_CODES[condition]
    cond_arr = np.asarray(subject["condition"])
    mask = np.isin(cond_arr, cond_codes)
    n_trials = int(mask.sum())

    if n_trials == 0:
        raise ValueError(
            f"No trials found for subject {subject['subject_id']} in "
            f"condition {condition} (codes {cond_codes}); expected >0"
        )

    bag = np.asarray(subject["outcome"])[mask].astype(np.float64)
    bucket = np.asarray(subject["prediction"])[mask].astype(np.float64)
    cond_seq = cond_arr[mask]

    # newBlock at trial 0 and at every condition-code transition within
    # this noise structure (avoid→seek boundary).  Matches MATLAB
    # getTrialVarsFromPEs_cannon.m:117 which resets RU at every newBlock.
    new_block_mask = np.zeros(n_trials, dtype=bool)
    new_block_mask[0] = True
    if n_trials > 1:
        new_block_mask[1:] = cond_seq[1:] != cond_seq[:-1]

    metadata = {
        "subject_id": str(subject["subject_id"]),
        "is_patient": bool(subject["is_patient"]),
        "n_trials": n_trials,
        "n_blocks": int(new_block_mask.sum()),
        "sub_blocks_avoid_seek": [int(c) for c in np.unique(cond_seq).tolist()],
    }
    return bag, bucket, new_block_mask, metadata


# ---------------------------------------------------------------------------
# Posterior predictive metrics
# ---------------------------------------------------------------------------


def compute_behavioral_metrics(
    posterior_samples: dict, bucket_positions: np.ndarray
) -> dict[str, float]:
    """Compute observed-vs-predicted bucket-update fit metrics.

    Uses the posterior mean of the deterministic ``normative_update``
    site as the model's per-trial prediction, then computes Pearson r,
    RMSE and MAE against the observed bucket update.

    Parameters
    ----------
    posterior_samples : dict
        Output of ``mcmc.get_samples()`` — must contain
        ``"normative_update"`` (n_samples, n_trials).
    bucket_positions : np.ndarray
        Observed bucket positions, shape ``(n_trials,)``.

    Returns
    -------
    dict[str, float]
        ``{"r_obs_vs_pred", "rmse", "mae", "n_trials"}``.

    Notes
    -----
    The first observed update is by convention zero (``np.diff`` with
    prepend of the first position) and is excluded from metrics so the
    correlation is not artificially inflated.
    """
    if "normative_update" not in posterior_samples:
        return {"r_obs_vs_pred": float("nan"), "rmse": float("nan"),
                "mae": float("nan"), "n_trials": 0}

    norm_update_samples = np.asarray(posterior_samples["normative_update"])
    predicted_mean = np.mean(norm_update_samples, axis=0)

    observed_update = np.diff(bucket_positions, prepend=bucket_positions[0])

    # Exclude the first trial (forced-zero observed update by prepend convention)
    obs = observed_update[1:]
    pred = predicted_mean[1:]
    if len(obs) < 2:
        return {"r_obs_vs_pred": float("nan"), "rmse": float("nan"),
                "mae": float("nan"), "n_trials": int(len(obs))}

    r_val, _ = pearsonr(obs, pred)
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    mae = float(np.mean(np.abs(obs - pred)))

    return {
        "r_obs_vs_pred": float(r_val),
        "rmse": rmse,
        "mae": mae,
        "n_trials": int(len(obs)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Fit one (subject_idx, condition) pair and write outputs."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    per_fit_dir = output_dir / "per_fit"
    per_fit_dir.mkdir(parents=True, exist_ok=True)

    # ----- Data -----
    data_path = Path(args.data_path)
    bag, bucket, new_block_mask, meta = load_subject_data(
        data_path, args.subject_idx, args.condition
    )

    print(
        f"Subject {args.subject_idx:3d} | id={meta['subject_id']} | "
        f"patient={meta['is_patient']} | condition={args.condition} | "
        f"n_trials={meta['n_trials']} | n_blocks={meta['n_blocks']}"
    )

    # ----- Seed -----
    cond_idx = 0 if args.condition == "changepoint" else 1
    fit_seed = args.seed + args.subject_idx * 100 + cond_idx

    # ----- MCMC -----
    start_t = time.time()
    mcmc, status, attempts = fit_with_retry(
        reduced_bayesian_model,
        {
            "bag_positions": jnp.asarray(bag),
            "bucket_positions": jnp.asarray(bucket),
            "context": args.condition,
            "new_block": jnp.asarray(new_block_mask, dtype=jnp.bool_),
        },
        seed=fit_seed,
        num_warmup_first=args.num_warmup,
        num_warmup_retry=args.num_warmup * 2,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
    )
    elapsed = time.time() - start_t

    if status == "FAILED":
        print(
            f"  WARNING: subject {meta['subject_id']}/{args.condition} fit FAILED "
            f"(R-hat or ESS gate not met); using posterior anyway"
        )

    # ----- Fit summary -----
    fit_json = make_fit_summary(
        mcmc,
        status=status,
        attempts=attempts,
        var_names=PARAM_NAMES,
        dataset_idx=int(args.subject_idx),
        condition=args.condition,
    )

    # ----- Posterior samples (raw + deterministics) -----
    posterior_samples = mcmc.get_samples()

    # ----- Behavioral metrics -----
    behavior = compute_behavioral_metrics(posterior_samples, bucket)
    print(
        f"  fit elapsed={elapsed:.1f}s | status={status} | "
        f"r_obs_vs_pred={behavior['r_obs_vs_pred']:.3f} | "
        f"RMSE={behavior['rmse']:.2f} | MAE={behavior['mae']:.2f}"
    )

    # ----- Output JSON (summary) -----
    out_summary = {
        "subject_idx": int(args.subject_idx),
        "subject_id": meta["subject_id"],
        "is_patient": meta["is_patient"],
        "condition": args.condition,
        "n_trials": meta["n_trials"],
        "n_blocks": meta["n_blocks"],
        "sub_blocks_avoid_seek": meta["sub_blocks_avoid_seek"],
        "mcmc_config": {
            "num_warmup": int(args.num_warmup),
            "num_samples": int(args.num_samples),
            "num_chains": int(args.num_chains),
        },
        "fit": fit_json,
        "behavior": behavior,
        "elapsed_seconds": float(elapsed),
    }
    json_path = per_fit_dir / f"{meta['subject_id']}_{args.condition}.json"
    with open(json_path, "w") as f:
        json.dump(to_jsonable(out_summary), f, indent=2)

    # ----- Output NPZ (raw posterior samples + deterministic posterior means) -----
    npz_path = per_fit_dir / f"{meta['subject_id']}_{args.condition}.npz"
    npz_data: dict[str, np.ndarray] = {}
    # Raw samples for the 5 fitted params + UU deterministic
    for k in PARAM_NAMES + ["UU"]:
        if k in posterior_samples:
            npz_data[f"posterior_{k}"] = np.asarray(posterior_samples[k])
    # Posterior MEAN of trial-level deterministics (saving raw arrays for all
    # 8000+ samples × 388 trials × 4 sites would balloon files; mean+5/95
    # quantiles is sufficient for downstream behavioral analyses)
    for k in DETERMINISTIC_NAMES:
        if k in posterior_samples:
            arr = np.asarray(posterior_samples[k])
            npz_data[f"{k}_mean"] = np.mean(arr, axis=0)
            npz_data[f"{k}_q05"] = np.quantile(arr, 0.05, axis=0)
            npz_data[f"{k}_q95"] = np.quantile(arr, 0.95, axis=0)
    npz_data["bag_positions"] = bag
    npz_data["bucket_positions"] = bucket
    npz_data["new_block_mask"] = new_block_mask
    np.savez_compressed(npz_path, **npz_data)

    print(f"  wrote {json_path}")
    print(f"  wrote {npz_path}")


if __name__ == "__main__":
    main()
