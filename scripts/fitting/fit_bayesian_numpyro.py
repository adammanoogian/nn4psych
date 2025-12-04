#!/usr/bin/env python3
"""
Bayesian Model Fitting using NumPyro

This script fits Bayesian normative models using NumPyro for full Bayesian
inference via MCMC. Provides posterior distributions for all parameters
and uncertainty quantification.

Advantages over PyEM:
- Full posterior distributions (not just point estimates)
- Credible intervals for all parameters
- Posterior predictive checks
- Model comparison via WAIC

Usage:
    python fit_bayesian_numpyro.py data.npy --context changepoint --num-samples 2000
    python fit_bayesian_numpyro.py data.npy --context oddball --num-chains 8

Reference:
    Loosen et al. (2023) - https://link.springer.com/article/10.3758/s13428-024-02427-y
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bayesian.numpyro_models import (
    run_mcmc,
    summarize_posterior,
    posterior_predictive,
    compute_waic,
    get_map_estimate,
)
from config import OUTPUT_DIR, BEHAVIORAL_FIGURES_DIR


def load_behavioral_data(data_path: str) -> tuple:
    """
    Load behavioral data from file.

    Parameters
    ----------
    data_path : str
        Path to behavioral data file (.npy format).

    Returns
    -------
    tuple
        (bucket_positions, bag_positions)
    """
    states = np.load(data_path)
    bucket_positions = states[1]  # index 1: bucket positions
    bag_positions = states[2]      # index 2: bag positions

    print(f"Loaded data from: {data_path}")
    print(f"  Bucket positions shape: {bucket_positions.shape}")
    print(f"  Bag positions shape: {bag_positions.shape}")

    return bucket_positions, bag_positions


def fit_numpyro_model(
    bucket_positions: np.ndarray,
    bag_positions: np.ndarray,
    context: str = 'changepoint',
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    seed: int = 42,
    save_results: bool = True,
) -> dict:
    """
    Fit NumPyro Bayesian model to behavioral data.

    Parameters
    ----------
    bucket_positions : np.ndarray
        Bucket position trajectory
    bag_positions : np.ndarray
        Bag position trajectory
    context : str
        Context: 'changepoint' or 'oddball'
    num_warmup : int
        Number of warmup iterations (default: 1000)
    num_samples : int
        Number of posterior samples per chain (default: 2000)
    num_chains : int
        Number of MCMC chains (default: 4)
    seed : int
        Random seed (default: 42)
    save_results : bool
        Whether to save fitting results

    Returns
    -------
    dict
        Fitted results including MCMC object and summaries
    """
    print(f"\nFitting NumPyro model for {context} condition...")
    print(f"MCMC settings:")
    print(f"  Warmup iterations: {num_warmup}")
    print(f"  Samples per chain: {num_samples}")
    print(f"  Number of chains: {num_chains}")
    print(f"  Total posterior samples: {num_samples * num_chains}")

    # Run MCMC
    mcmc = run_mcmc(
        bucket_positions,
        bag_positions,
        context=context,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        progress_bar=True,
    )

    # Print diagnostics
    print("\n" + "="*60)
    print("MCMC DIAGNOSTICS")
    print("="*60)
    mcmc.print_summary()

    # Get posterior summary
    posterior_summary = summarize_posterior(mcmc, prob=0.89)

    print("\n" + "="*60)
    print("POSTERIOR SUMMARY (89% HPDI)")
    print("="*60)
    for param, stats in posterior_summary.items():
        print(f"{param:12s}: {stats['mean']:.3f} ± {stats['std']:.3f}  "
              f"[{stats['hpdi_low']:.3f}, {stats['hpdi_high']:.3f}]")

    # Get MAP estimate for comparison with PyEM
    map_estimate = get_map_estimate(mcmc)
    print("\n" + "="*60)
    print("MAP ESTIMATES (for comparison with PyEM)")
    print("="*60)
    for param, value in map_estimate.items():
        print(f"{param:12s}: {value:.3f}")

    # Compute WAIC
    try:
        waic_stats = compute_waic(mcmc, bucket_positions, bag_positions, context)
        print("\n" + "="*60)
        print("MODEL COMPARISON METRICS")
        print("="*60)
        print(f"WAIC: {waic_stats['waic']:.2f} ± {waic_stats['se']:.2f}")
        print(f"Effective parameters (p_WAIC): {waic_stats['p_waic']:.2f}")
    except Exception as e:
        print(f"\nWarning: Could not compute WAIC: {e}")
        waic_stats = None

    # Generate posterior predictive samples
    print("\nGenerating posterior predictive samples...")
    posterior_pred = posterior_predictive(
        mcmc,
        bucket_positions,
        bag_positions,
        context,
        num_samples=500,
    )

    results = {
        'context': context,
        'mcmc': mcmc,
        'posterior_summary': posterior_summary,
        'map_estimate': map_estimate,
        'waic': waic_stats,
        'posterior_predictive': posterior_pred,
        'bucket_positions': bucket_positions,
        'bag_positions': bag_positions,
    }

    if save_results:
        output_dir = OUTPUT_DIR / 'bayesian_fits'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results (excluding MCMC object which can be large)
        results_to_save = {
            k: v for k, v in results.items()
            if k not in ['mcmc', 'posterior_predictive']
        }

        results_path = output_dir / f'numpyro_results_{context}.npy'
        np.save(results_path, results_to_save, allow_pickle=True)
        print(f"\nSaved results to: {results_path}")

        # Save posterior samples separately
        samples = mcmc.get_samples()
        samples_np = {k: np.array(v) for k, v in samples.items()}
        samples_path = output_dir / f'numpyro_posterior_samples_{context}.npy'
        np.save(samples_path, samples_np)
        print(f"Saved posterior samples to: {samples_path}")

    print("\n" + "="*60)

    return results


def plot_posterior_distributions(
    results: dict,
    save_path: Path = None,
):
    """
    Plot posterior distributions for all parameters.

    Parameters
    ----------
    results : dict
        Results from fit_numpyro_model
    save_path : Path, optional
        Path to save figure
    """
    mcmc = results['mcmc']
    context = results['context']

    # Get samples
    samples = mcmc.get_samples()

    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    param_names = ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']
    param_labels = ['Hazard Rate (H)', 'Likelihood Weight (LW)',
                   'Uncertainty Underest. (UU)', 'Motor Variance (σ_motor)',
                   'LR Variance Slope (σ_LR)']

    for i, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[i]
        param_samples = np.array(samples[param])

        # Histogram
        ax.hist(param_samples, bins=40, alpha=0.6, color='steelblue',
               edgecolor='black', density=True)

        # Add vertical lines for mean and credible interval
        mean_val = param_samples.mean()
        summary = results['posterior_summary'][param]

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {mean_val:.3f}')
        ax.axvline(summary['hpdi_low'], color='orange', linestyle=':',
                  linewidth=1.5, alpha=0.7)
        ax.axvline(summary['hpdi_high'], color='orange', linestyle=':',
                  linewidth=1.5, alpha=0.7, label=f"89% HPDI")

        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide the extra subplot
    axes[5].axis('off')

    plt.suptitle(f'Posterior Distributions: {context.capitalize()}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved posterior distributions plot: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_trace(results: dict, save_path: Path = None):
    """
    Plot MCMC trace plots for diagnostics.

    Parameters
    ----------
    results : dict
        Results from fit_numpyro_model
    save_path : Path, optional
        Path to save figure
    """
    mcmc = results['mcmc']

    # Convert to ArviZ InferenceData for nice plotting
    idata = az.from_numpyro(mcmc)

    # Create trace plot
    az.plot_trace(idata, var_names=['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR'],
                 compact=False, figsize=(12, 10))

    plt.suptitle(f"MCMC Trace Plots: {results['context'].capitalize()}",
                fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trace plot: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_posterior_predictive_check(results: dict, save_path: Path = None):
    """
    Plot posterior predictive check.

    Parameters
    ----------
    results : dict
        Results from fit_numpyro_model
    save_path : Path, optional
        Path to save figure
    """
    posterior_pred = results['posterior_predictive']
    bucket_positions = results['bucket_positions']

    # Compute observed bucket updates
    bucket_update_obs = np.diff(bucket_positions, prepend=bucket_positions[0])

    # Get posterior predictive samples (shape: [num_samples, n_trials])
    bucket_update_pred = posterior_pred['bucket_update']

    # Compute statistics
    pred_mean = bucket_update_pred.mean(axis=0)
    pred_std = bucket_update_pred.std(axis=0)
    pred_5 = np.percentile(bucket_update_pred, 5, axis=0)
    pred_95 = np.percentile(bucket_update_pred, 95, axis=0)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    trials = np.arange(len(bucket_update_obs))

    # Panel 1: Time series
    ax1.fill_between(trials, pred_5, pred_95, alpha=0.3, color='steelblue',
                     label='90% Pred. Interval')
    ax1.plot(trials, pred_mean, 'b-', linewidth=2, label='Posterior Mean')
    ax1.scatter(trials, bucket_update_obs, c='red', s=30, alpha=0.6,
               label='Observed', zorder=10)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Bucket Update')
    ax1.set_title('Posterior Predictive Check: Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Observed vs Predicted
    ax2.scatter(pred_mean, bucket_update_obs, alpha=0.5, s=40, c='steelblue')
    ax2.errorbar(pred_mean, bucket_update_obs, xerr=pred_std, fmt='none',
                alpha=0.2, color='gray')

    # Identity line
    min_val = min(pred_mean.min(), bucket_update_obs.min())
    max_val = max(pred_mean.max(), bucket_update_obs.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5,
            alpha=0.5, label='Perfect Prediction')

    # Correlation
    corr = np.corrcoef(pred_mean, bucket_update_obs)[0, 1]
    ax2.set_xlabel('Predicted Update (Posterior Mean)')
    ax2.set_ylabel('Observed Update')
    ax2.set_title(f'Posterior Predictive Check (r = {corr:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Posterior Predictive Check: {results['context'].capitalize()}",
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved posterior predictive check: {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fit Bayesian normative model using NumPyro (MCMC)"
    )
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to behavioral data (.npy file)',
    )
    parser.add_argument(
        '--context',
        type=str,
        default='changepoint',
        choices=['changepoint', 'oddball'],
        help='Task context (default: changepoint)',
    )
    parser.add_argument(
        '--num-warmup',
        type=int,
        default=1000,
        help='Number of warmup iterations (default: 1000)',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=2000,
        help='Number of samples per chain (default: 2000)',
    )
    parser.add_argument(
        '--num-chains',
        type=int,
        default=4,
        help='Number of MCMC chains (default: 4)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results',
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Do not generate plots',
    )

    args = parser.parse_args()

    # Load data
    bucket_positions, bag_positions = load_behavioral_data(args.data_path)

    # Fit model
    results = fit_numpyro_model(
        bucket_positions,
        bag_positions,
        context=args.context,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        save_results=not args.no_save,
    )

    # Generate plots
    if not args.no_plot:
        plots_dir = BEHAVIORAL_FIGURES_DIR
        plots_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating plots...")

        # Posterior distributions
        plot_posterior_distributions(
            results,
            save_path=plots_dir / f'numpyro_posterior_{args.context}.png'
        )

        # Trace plots
        plot_trace(
            results,
            save_path=plots_dir / f'numpyro_trace_{args.context}.png'
        )

        # Posterior predictive check
        plot_posterior_predictive_check(
            results,
            save_path=plots_dir / f'numpyro_ppc_{args.context}.png'
        )

    print("\nFitting complete!")


if __name__ == "__main__":
    main()
