#!/usr/bin/env python3
"""
Bayesian Model Fitting using PyEM

This script fits normative models to behavioral data using the PyEM framework
(Loosen et al., 2023). Implements parameter estimation for changepoint and
oddball conditions.

Reference:
Loosen et al. (2023) https://link.springer.com/article/10.3758/s13428-024-02427-y

Refactored to use nn4psych package structure.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nn4psych.models.bayesian import norm2beta, norm2alpha
from nn4psych.models.bayesian.pyem_models import fit
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


def fit_pyem_model(
    bucket_positions: np.ndarray,
    bag_positions: np.ndarray,
    context: str = 'changepoint',
    initial_params: np.ndarray = None,
    save_results: bool = True,
) -> dict:
    """
    Fit PyEM normative model to behavioral data.

    Parameters
    ----------
    bucket_positions : np.ndarray
        Bucket position trajectory.
    bag_positions : np.ndarray
        Bag position trajectory.
    context : str
        Context: 'changepoint' or 'oddball'.
    initial_params : np.ndarray, optional
        Initial parameter values. If None, uses defaults.
    save_results : bool
        Whether to save fitting results.

    Returns
    -------
    dict
        Fitted parameters and model outputs.
    """
    print(f"\nFitting PyEM model for {context} condition...")

    # Set initial parameters if not provided
    if initial_params is None:
        # Default initial values (in normalized space)
        initial_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        print("Using default initial parameters")
    else:
        print(f"Using provided initial parameters: {initial_params}")

    # Optimize parameters
    result = minimize(
        fit,
        initial_params,
        args=(bucket_positions, bag_positions, context, None, 'nll'),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'disp': True}
    )

    # Transform parameters back to interpretable space
    fitted_params = result.x
    H = norm2alpha(fitted_params[0])           # Hazard rate
    LW = norm2alpha(fitted_params[1])          # Likelihood weight
    UU = norm2alpha(fitted_params[2])          # Uncertainty underestimation
    sigma_motor = norm2beta(fitted_params[3])  # Update variance
    sigma_LR = norm2beta(fitted_params[4])     # Update variance slope

    print("\n" + "="*60)
    print("FITTED PARAMETERS")
    print("="*60)
    print(f"  Hazard rate (H):              {H:.4f}")
    print(f"  Likelihood weight (LW):       {LW:.4f}")
    print(f"  Uncertainty underest. (UU):   {UU:.4f}")
    print(f"  Motor variance (σ_motor):     {sigma_motor:.4f}")
    print(f"  LR variance slope (σ_LR):     {sigma_LR:.4f}")
    print(f"  Negative log-likelihood:      {result.fun:.2f}")
    print("="*60)

    # Get full model outputs
    model_outputs = fit(
        fitted_params,
        bucket_positions,
        bag_positions,
        context,
        prior=None,
        output='all'
    )

    results = {
        'context': context,
        'fitted_params_raw': fitted_params,
        'fitted_params': {
            'H': H,
            'LW': LW,
            'UU': UU,
            'sigma_motor': sigma_motor,
            'sigma_LR': sigma_LR,
        },
        'negll': result.fun,
        'optimization_result': result,
        'model_outputs': model_outputs,
    }

    if save_results:
        output_dir = OUTPUT_DIR / 'bayesian_fits'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save parameters
        param_path = output_dir / f'pyem_params_{context}.npy'
        np.save(param_path, results['fitted_params'])
        print(f"\nSaved parameters to: {param_path}")

        # Save full results
        results_path = output_dir / f'pyem_full_results_{context}.npy'
        np.save(results_path, results)
        print(f"Saved full results to: {results_path}")

    return results


def plot_fit_results(
    results: dict,
    bucket_positions: np.ndarray,
    bag_positions: np.ndarray,
):
    """
    Visualize fitting results.

    Parameters
    ----------
    results : dict
        Fitting results from fit_pyem_model.
    bucket_positions : np.ndarray
        Actual bucket positions.
    bag_positions : np.ndarray
        Actual bag positions.
    """
    context = results['context']
    model_outputs = results['model_outputs']

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    n_trials = len(bucket_positions)
    trials = np.arange(n_trials)

    # Plot 1: Positions
    ax = axes[0, 0]
    ax.plot(trials, bucket_positions, label='Bucket (actual)', color='orange', linewidth=2)
    ax.plot(trials, model_outputs['pred_bucket_placement'],
            label='Bucket (predicted)', color='blue', linestyle='--', linewidth=2)
    ax.scatter(trials, bag_positions, label='Bag', color='red', alpha=0.5, s=20)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Position')
    ax.set_title('Actual vs Predicted Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Learning Rate
    ax = axes[0, 1]
    ax.plot(trials, model_outputs['learning_rate'], color='steelblue', linewidth=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Model Learning Rate')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Plot 3: Prediction Error
    ax = axes[1, 0]
    ax.plot(trials, model_outputs['pred_error'], color='crimson', linewidth=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Prediction Error')
    ax.grid(True, alpha=0.3)

    # Plot 4: Changepoint Probability (Omega)
    ax = axes[1, 1]
    ax.plot(trials, model_outputs['omega'], color='purple', linewidth=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Changepoint Probability (Ω)')
    ax.set_title('Changepoint Probability')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Plot 5: Relative Uncertainty (Tau)
    ax = axes[2, 0]
    ax.plot(trials, model_outputs['tau'][:-1], color='teal', linewidth=2)  # tau has n+1 elements
    ax.set_xlabel('Trial')
    ax.set_ylabel('Relative Uncertainty (τ)')
    ax.set_title('Relative Uncertainty')
    ax.grid(True, alpha=0.3)

    # Plot 6: Update comparison
    ax = axes[2, 1]
    ax.plot(trials, model_outputs['bucket_update'],
            label='Actual Update', color='orange', linewidth=2, alpha=0.7)
    ax.plot(trials, model_outputs['normative_update'],
            label='Normative Update', color='blue', linewidth=2, alpha=0.7)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Update')
    ax.set_title('Actual vs Normative Update')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'PyEM Model Fit: {context.capitalize()}\n' +
                 f'H={results["fitted_params"]["H"]:.3f}, ' +
                 f'LW={results["fitted_params"]["LW"]:.3f}, ' +
                 f'UU={results["fitted_params"]["UU"]:.3f}',
                 fontsize=14)
    plt.tight_layout()

    # Save figure
    output_dir = BEHAVIORAL_FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'pyem_fit_{context}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fit visualization: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fit PyEM normative model to behavioral data"
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
        help='Task context',
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
    results = fit_pyem_model(
        bucket_positions,
        bag_positions,
        context=args.context,
        save_results=not args.no_save,
    )

    # Visualize results
    if not args.no_plot:
        plot_fit_results(results, bucket_positions, bag_positions)

    print("\nFitting complete!")


if __name__ == "__main__":
    main()
