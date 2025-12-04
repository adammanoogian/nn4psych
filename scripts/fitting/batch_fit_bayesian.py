#!/usr/bin/env python3
"""
Batch Bayesian Model Fitting

This script fits Bayesian normative models to multiple datasets in parallel.
Supports both changepoint and oddball conditions with comprehensive output.

Usage:
    python batch_fit_bayesian.py data_dir/ --output results/ --context changepoint
    python batch_fit_bayesian.py data_dir/*.npy --output results/ --parallel

Reference:
    Loosen et al. (2023) - https://link.springer.com/article/10.3758/s13428-024-02427-y
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.optimize import minimize
from tqdm import tqdm
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bayesian import fit_bayesian_model, norm2alpha, norm2beta
from bayesian.model_comparison import calculate_bic, calculate_aic
from bayesian.visualization import (
    plot_model_fit_comprehensive,
    plot_parameter_distributions
)
from config import OUTPUT_DIR, BEHAVIORAL_FIGURES_DIR


def load_data_file(data_path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load behavioral data from a file.

    Parameters
    ----------
    data_path : Path
        Path to data file (.npy format)

    Returns
    -------
    tuple
        (bucket_positions, bag_positions, dataset_name)
    """
    states = np.load(data_path)

    # Handle different data formats
    if states.ndim == 1:
        raise ValueError(f"Invalid data shape in {data_path}: expected 2D array")
    elif states.shape[0] == 3:
        # Format: [trials, bucket, bag]
        bucket_positions = states[1]
        bag_positions = states[2]
    elif states.shape[0] >= 5:
        # Format: [trials, bucket, bag, helicopter, hazard]
        bucket_positions = states[1]
        bag_positions = states[2]
    else:
        raise ValueError(f"Unexpected data format in {data_path}: shape {states.shape}")

    dataset_name = data_path.stem

    return bucket_positions, bag_positions, dataset_name


def fit_single_dataset(
    bucket_positions: np.ndarray,
    bag_positions: np.ndarray,
    dataset_name: str,
    context: str,
    initial_params: np.ndarray = None,
    verbose: bool = True,
) -> Dict:
    """
    Fit Bayesian model to a single dataset.

    Parameters
    ----------
    bucket_positions : np.ndarray
        Bucket position trajectory
    bag_positions : np.ndarray
        Bag position trajectory
    dataset_name : str
        Identifier for this dataset
    context : str
        'changepoint' or 'oddball'
    initial_params : np.ndarray, optional
        Initial parameter values (default: all zeros)
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        Fitting results including parameters and metrics
    """
    if initial_params is None:
        initial_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    if verbose:
        print(f"Fitting {dataset_name} ({context})...")

    # Optimize parameters
    result = minimize(
        fit_bayesian_model,
        initial_params,
        args=(bucket_positions, bag_positions, context, None, 'nll'),
        method='Nelder-Mead',
        options={'maxiter': 10000}
    )

    # Transform parameters
    fitted_params = result.x
    H = norm2alpha(fitted_params[0])
    LW = norm2alpha(fitted_params[1])
    UU = norm2alpha(fitted_params[2])
    sigma_motor = norm2beta(fitted_params[3])
    sigma_LR = norm2beta(fitted_params[4])

    # Get model outputs
    model_outputs = fit_bayesian_model(
        fitted_params,
        bucket_positions,
        bag_positions,
        context,
        prior=None,
        output='all'
    )

    # Calculate information criteria
    n_trials = len(bucket_positions)
    n_params = len(fitted_params)
    negll = result.fun
    bic = calculate_bic(negll, n_params, n_trials)
    aic = calculate_aic(negll, n_params)

    # Calculate fit quality metrics
    residuals = model_outputs['bucket_update'] - model_outputs['normative_update']
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    r_corr = np.corrcoef(
        model_outputs['normative_update'],
        model_outputs['bucket_update']
    )[0, 1]

    if verbose:
        print(f"  H={H:.3f}, LW={LW:.3f}, UU={UU:.3f}")
        print(f"  NegLL={negll:.2f}, BIC={bic:.2f}, R={r_corr:.3f}")

    return {
        'dataset': dataset_name,
        'context': context,
        'n_trials': n_trials,
        'H': H,
        'LW': LW,
        'UU': UU,
        'sigma_motor': sigma_motor,
        'sigma_LR': sigma_LR,
        'fitted_params_raw': fitted_params,
        'negll': negll,
        'BIC': bic,
        'AIC': aic,
        'RMSE': rmse,
        'MAE': mae,
        'R': r_corr,
        'R_squared': r_corr**2,
        'convergence': result.success,
        'n_iterations': result.nit,
        'model_outputs': model_outputs,
        'bucket_positions': bucket_positions,
        'bag_positions': bag_positions,
    }


def batch_fit_datasets(
    data_paths: List[Path],
    context: str,
    output_dir: Path,
    save_plots: bool = True,
    parallel: bool = False,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Fit Bayesian models to multiple datasets.

    Parameters
    ----------
    data_paths : list of Path
        Paths to data files
    context : str
        'changepoint' or 'oddball'
    output_dir : Path
        Directory to save results
    save_plots : bool
        Whether to save individual plots
    parallel : bool
        Whether to use parallel processing
    n_jobs : int
        Number of parallel jobs (-1 for all cores)

    Returns
    -------
    pd.DataFrame
        Summary of all fits
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    if save_plots:
        plots_dir.mkdir(exist_ok=True)

    all_results = []

    if parallel:
        from joblib import Parallel, delayed
        print(f"Fitting {len(data_paths)} datasets in parallel...")

        def fit_one(path):
            bucket, bag, name = load_data_file(path)
            return fit_single_dataset(bucket, bag, name, context, verbose=False)

        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_one)(path) for path in tqdm(data_paths)
        )
        all_results = results
    else:
        print(f"Fitting {len(data_paths)} datasets sequentially...")
        for path in tqdm(data_paths):
            bucket, bag, name = load_data_file(path)
            result = fit_single_dataset(bucket, bag, name, context, verbose=False)
            all_results.append(result)

    # Create summary DataFrame
    summary_data = []
    for res in all_results:
        summary_data.append({
            'dataset': res['dataset'],
            'context': res['context'],
            'n_trials': res['n_trials'],
            'H': res['H'],
            'LW': res['LW'],
            'UU': res['UU'],
            'sigma_motor': res['sigma_motor'],
            'sigma_LR': res['sigma_LR'],
            'negll': res['negll'],
            'BIC': res['BIC'],
            'AIC': res['AIC'],
            'RMSE': res['RMSE'],
            'MAE': res['MAE'],
            'R': res['R'],
            'R_squared': res['R_squared'],
            'convergence': res['convergence'],
            'n_iterations': res['n_iterations'],
        })

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_path = output_dir / f'summary_{context}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")

    # Save full results
    full_results_path = output_dir / f'full_results_{context}.npy'
    np.save(full_results_path, all_results)
    print(f"Saved full results to: {full_results_path}")

    # Generate summary plots
    if len(all_results) > 1:
        print("\nGenerating summary visualizations...")

        # Parameter distributions
        param_estimates = {
            'H': np.array([r['H'] for r in all_results]),
            'LW': np.array([r['LW'] for r in all_results]),
            'UU': np.array([r['UU'] for r in all_results]),
            'σ_motor': np.array([r['sigma_motor'] for r in all_results]),
            'σ_LR': np.array([r['sigma_LR'] for r in all_results]),
        }

        fig = plot_parameter_distributions(
            param_estimates,
            save_path=output_dir / f'param_distributions_{context}.png'
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    # Save individual plots if requested
    if save_plots:
        print("\nGenerating individual fit plots...")
        for res in tqdm(all_results, desc="Plotting"):
            plot_path = plots_dir / f"{res['dataset']}_{context}.png"
            fig = plot_model_fit_comprehensive(
                res['model_outputs'],
                res['bucket_positions'],
                res['bag_positions'],
                save_path=plot_path
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

    # Print summary statistics
    print("\n" + "="*60)
    print("BATCH FITTING SUMMARY")
    print("="*60)
    print(f"Context: {context}")
    print(f"Datasets fitted: {len(all_results)}")
    print(f"Successful convergence: {summary_df['convergence'].sum()}/{len(summary_df)}")
    print("\nParameter Estimates (Mean ± SD):")
    for param in ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']:
        mean = summary_df[param].mean()
        std = summary_df[param].std()
        print(f"  {param:12s}: {mean:.3f} ± {std:.3f}")
    print(f"\nModel Fit (Mean ± SD):")
    print(f"  NegLL:  {summary_df['negll'].mean():.2f} ± {summary_df['negll'].std():.2f}")
    print(f"  BIC:    {summary_df['BIC'].mean():.2f} ± {summary_df['BIC'].std():.2f}")
    print(f"  RMSE:   {summary_df['RMSE'].mean():.3f} ± {summary_df['RMSE'].std():.3f}")
    print(f"  R²:     {summary_df['R_squared'].mean():.3f} ± {summary_df['R_squared'].std():.3f}")
    print("="*60)

    return summary_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch fit Bayesian normative models to multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'data',
        type=str,
        nargs='+',
        help='Data files or directory containing .npy files',
    )
    parser.add_argument(
        '--context',
        type=str,
        default='changepoint',
        choices=['changepoint', 'oddball', 'both'],
        help='Task context to fit (default: changepoint)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: output/bayesian_fits/batch)',
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel processing',
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all cores)',
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Do not generate individual plots',
    )

    args = parser.parse_args()

    # Collect data files
    data_paths = []
    for item in args.data:
        path = Path(item)
        if path.is_dir():
            data_paths.extend(path.glob('*.npy'))
        elif path.is_file() and path.suffix == '.npy':
            data_paths.append(path)
        else:
            print(f"Warning: Skipping {item} (not a .npy file or directory)")

    if not data_paths:
        print("Error: No .npy files found!")
        return

    print(f"Found {len(data_paths)} data files")

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = OUTPUT_DIR / 'bayesian_fits' / 'batch'

    # Fit models
    contexts_to_fit = ['changepoint', 'oddball'] if args.context == 'both' else [args.context]

    for context in contexts_to_fit:
        print(f"\n{'='*60}")
        print(f"FITTING {context.upper()} MODEL")
        print(f"{'='*60}\n")

        summary = batch_fit_datasets(
            data_paths,
            context=context,
            output_dir=output_dir,
            save_plots=not args.no_plots,
            parallel=args.parallel,
            n_jobs=args.n_jobs,
        )

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
