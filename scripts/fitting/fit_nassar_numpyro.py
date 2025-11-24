#!/usr/bin/env python3
"""
Fit Bayesian Normative Model to Nassar et al. 2021 Data using NumPyro

This script fits the Bayesian normative model to extracted trial-by-trial data
from Nassar et al. (2021) using NumPyro MCMC for full Bayesian inference.

We fit each subject's data separately for changepoint and oddball contexts,
then compare our fitted parameters to Nassar's reference parameters.

Usage:
    # Fit all subjects
    python fit_nassar_numpyro.py

    # Fit specific subjects
    python fit_nassar_numpyro.py --subjects SP_063808 SP_066396

    # Quick test run (fewer samples)
    python fit_nassar_numpyro.py --quick
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bayesian.numpyro_models import run_mcmc, summarize_posterior, get_map_estimate
from config import OUTPUT_DIR

# Condition mapping (from extract_nassar_trials.py)
CONDITION_MAP = {
    0: 'cp_avoid',
    1: 'cp_seek',
    2: 'drift_avoid',
    3: 'drift_seek',
}

# Context mapping for Bayesian fitting
CONTEXT_MAP = {
    0: 'changepoint',  # cp_avoid
    1: 'changepoint',  # cp_seek
    2: 'oddball',      # drift_avoid
    3: 'oddball',      # drift_seek
}


def load_extracted_data():
    """Load extracted Nassar trial data."""
    data_path = OUTPUT_DIR / 'processed' / 'nassar2021' / 'subject_trials.npy'
    metadata_path = OUTPUT_DIR / 'processed' / 'nassar2021' / 'subject_metadata.csv'

    if not data_path.exists():
        raise FileNotFoundError(
            f"Extracted data not found at {data_path}\n"
            f"Run scripts/data_pipeline/extract_nassar_trials.py first"
        )

    print(f"Loading extracted data from: {data_path}")
    subjects_data = np.load(data_path, allow_pickle=True)
    metadata = pd.read_csv(metadata_path)

    print(f"Loaded {len(subjects_data)} subjects")
    print(f"  Patients: {metadata['is_patient'].sum()}")
    print(f"  Controls: {(~metadata['is_patient']).sum()}")

    return subjects_data, metadata


def prepare_subject_data(subject_data, context='changepoint'):
    """
    Prepare subject data for Bayesian fitting.

    Parameters
    ----------
    subject_data : dict
        Single subject's data from extracted file
    context : str
        'changepoint' or 'oddball'

    Returns
    -------
    bucket_positions : np.ndarray
        Bucket positions (predictions)
    bag_positions : np.ndarray
        Bag positions (outcomes)
    n_trials : int
        Number of trials for this context
    """
    # Filter by context
    conditions = subject_data['condition']

    if context == 'changepoint':
        # cp_avoid (0) and cp_seek (1)
        mask = (conditions == 0) | (conditions == 1)
    elif context == 'oddball':
        # drift_avoid (2) and drift_seek (3)
        mask = (conditions == 2) | (conditions == 3)
    else:
        raise ValueError(f"Unknown context: {context}")

    # Extract data
    bucket_positions = subject_data['prediction'][mask]
    bag_positions = subject_data['outcome'][mask]

    return bucket_positions, bag_positions, mask.sum()


def fit_subject(subject_data, context='changepoint',
                num_warmup=1000, num_samples=2000, num_chains=4,
                seed=42, verbose=False):
    """
    Fit Bayesian model to a single subject's data.

    Parameters
    ----------
    subject_data : dict
        Subject's trial data
    context : str
        'changepoint' or 'oddball'
    num_warmup : int
        MCMC warmup iterations
    num_samples : int
        Posterior samples per chain
    num_chains : int
        Number of MCMC chains
    seed : int
        Random seed
    verbose : bool
        Print detailed output

    Returns
    -------
    dict
        Fitted results including MAP estimates and posterior summary
    """
    subject_id = subject_data['subject_id']

    # Prepare data
    bucket_positions, bag_positions, n_trials = prepare_subject_data(
        subject_data, context
    )

    if verbose:
        print(f"\n  Fitting {context} context:")
        print(f"    Trials: {n_trials}")
        print(f"    Outcome range: [{bag_positions.min():.1f}, {bag_positions.max():.1f}]")
        print(f"    Running MCMC...")

    try:
        # Run MCMC
        mcmc = run_mcmc(
            bucket_positions,
            bag_positions,
            context=context,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=seed,
            progress_bar=verbose,
        )

        # Get posterior summary
        posterior_summary = summarize_posterior(mcmc, prob=0.89)

        # Get MAP estimate (for comparison with Nassar's point estimates)
        map_estimate = get_map_estimate(mcmc)

        if verbose:
            print(f"\n    MAP Estimates:")
            for param, value in map_estimate.items():
                print(f"      {param:12s}: {value:.3f}")

        return {
            'subject_id': subject_id,
            'context': context,
            'n_trials': n_trials,
            'map_estimate': map_estimate,
            'posterior_summary': posterior_summary,
            'converged': True,
            'error': None,
        }

    except Exception as e:
        print(f"\n    ERROR fitting {subject_id} ({context}): {e}")
        return {
            'subject_id': subject_id,
            'context': context,
            'n_trials': n_trials,
            'map_estimate': None,
            'posterior_summary': None,
            'converged': False,
            'error': str(e),
        }


def fit_all_subjects(subjects_data, metadata,
                     num_warmup=1000, num_samples=2000, num_chains=4,
                     quick=False, subject_ids=None):
    """
    Fit Bayesian models to all subjects.

    Parameters
    ----------
    subjects_data : list
        List of subject data dictionaries
    metadata : pd.DataFrame
        Subject metadata
    num_warmup : int
        MCMC warmup iterations
    num_samples : int
        Posterior samples per chain
    num_chains : int
        Number of chains
    quick : bool
        Quick test run with fewer samples
    subject_ids : list, optional
        List of specific subject IDs to fit

    Returns
    -------
    list
        Fitted results for all subjects
    """
    if quick:
        print("\n[QUICK MODE] Using reduced MCMC settings")
        num_warmup = 500
        num_samples = 1000
        num_chains = 2

    print(f"\n{'='*70}")
    print("FITTING BAYESIAN MODELS TO NASSAR DATA")
    print(f"{'='*70}")
    print(f"MCMC Settings:")
    print(f"  Warmup: {num_warmup}")
    print(f"  Samples per chain: {num_samples}")
    print(f"  Chains: {num_chains}")
    print(f"  Total posterior samples: {num_samples * num_chains}")

    # Filter subjects if requested
    if subject_ids is not None:
        subjects_data = [s for s in subjects_data if s['subject_id'] in subject_ids]
        print(f"\nFitting {len(subjects_data)} selected subjects")

    all_results = []

    # Fit each subject for both contexts
    for i, subject_data in enumerate(tqdm(subjects_data, desc="Fitting subjects")):
        subject_id = subject_data['subject_id']
        is_patient = subject_data['is_patient']

        print(f"\n[{i+1}/{len(subjects_data)}] {subject_id} ({'Patient' if is_patient else 'Control'})")

        # Fit changepoint context
        cp_result = fit_subject(
            subject_data,
            context='changepoint',
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=42 + i * 2,  # Different seed per subject
            verbose=False,
        )
        all_results.append(cp_result)

        # Fit oddball context
        ob_result = fit_subject(
            subject_data,
            context='oddball',
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=42 + i * 2 + 1,
            verbose=False,
        )
        all_results.append(ob_result)

    print(f"\n{'='*70}")
    print(f"Fitting complete!")
    print(f"  Total fits: {len(all_results)}")
    print(f"  Successful: {sum(1 for r in all_results if r['converged'])}")
    print(f"  Failed: {sum(1 for r in all_results if not r['converged'])}")
    print(f"{'='*70}")

    return all_results


def save_fitted_parameters(results, metadata):
    """
    Save fitted parameters to CSV for easy comparison.

    Parameters
    ----------
    results : list
        Fitted results from fit_all_subjects
    metadata : pd.DataFrame
        Subject metadata

    Returns
    -------
    pd.DataFrame
        Parameter table
    """
    rows = []

    for result in results:
        if not result['converged']:
            continue

        subject_id = result['subject_id']
        context = result['context']
        map_est = result['map_estimate']

        # Get patient status from metadata
        is_patient = metadata[metadata['subject_id'] == subject_id]['is_patient'].values[0]

        row = {
            'subject_id': subject_id,
            'is_patient': is_patient,
            'context': context,
            'n_trials': result['n_trials'],
            'H': map_est['H'],
            'LW': map_est['LW'],
            'UU': map_est['UU'],
            'sigma_motor': map_est['sigma_motor'],
            'sigma_LR': map_est['sigma_LR'],
        }

        rows.append(row)

    params_df = pd.DataFrame(rows)

    # Save to CSV
    output_dir = OUTPUT_DIR / 'fitted_params' / 'nassar2021'
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'numpyro_fitted_params.csv'
    params_df.to_csv(csv_path, index=False)
    print(f"\nSaved fitted parameters to: {csv_path}")

    # Also save full results
    results_path = output_dir / 'numpyro_full_results.npy'
    np.save(results_path, results, allow_pickle=True)
    print(f"Saved full results to: {results_path}")

    return params_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fit Bayesian models to Nassar et al. 2021 data using NumPyro"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run with reduced MCMC samples'
    )
    parser.add_argument(
        '--num-warmup',
        type=int,
        default=1000,
        help='Number of warmup iterations (default: 1000)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=2000,
        help='Number of samples per chain (default: 2000)'
    )
    parser.add_argument(
        '--num-chains',
        type=int,
        default=4,
        help='Number of MCMC chains (default: 4)'
    )
    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Specific subject IDs to fit (optional)'
    )

    args = parser.parse_args()

    # Load extracted data
    subjects_data, metadata = load_extracted_data()

    # Fit all subjects
    results = fit_all_subjects(
        subjects_data,
        metadata,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        quick=args.quick,
        subject_ids=args.subjects,
    )

    # Save results
    params_df = save_fitted_parameters(results, metadata)

    # Print summary statistics
    print("\n[SUMMARY STATISTICS]")
    print("\nChangepoint Context:")
    cp_params = params_df[params_df['context'] == 'changepoint']
    print(cp_params[['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']].describe())

    print("\nOddball Context:")
    ob_params = params_df[params_df['context'] == 'oddball']
    print(ob_params[['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']].describe())

    print("\n[PATIENT vs CONTROL]")
    for context in ['changepoint', 'oddball']:
        print(f"\n{context.capitalize()}:")
        context_data = params_df[params_df['context'] == context]

        patients = context_data[context_data['is_patient']]
        controls = context_data[~context_data['is_patient']]

        for param in ['H', 'LW', 'UU']:
            p_mean = patients[param].mean()
            c_mean = controls[param].mean()
            print(f"  {param:12s}: Patients={p_mean:.3f}, Controls={c_mean:.3f}")


if __name__ == "__main__":
    main()
