#!/usr/bin/env python3
"""
Bayesian Model Fitting using PyMC

This script fits Bayesian normative models to behavioral data using PyMC-based
Bayesian inference. It supports both MLE and full Bayesian posterior estimation.

Refactored to use nn4psych package structure.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bayesian.bayesian_models import BayesianModel
from config import OUTPUT_DIR, BEHAVIORAL_FIGURES_DIR


def load_behavioral_data(data_path: str) -> np.ndarray:
    """
    Load behavioral data from file.

    Parameters
    ----------
    data_path : str
        Path to behavioral data file (.npy format).

    Returns
    -------
    np.ndarray
        States array with shape [5, n_trials]:
        [trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers]
    """
    states = np.load(data_path)
    print(f"Loaded data from: {data_path}")
    print(f"  Shape: {states.shape}")
    return states


def fit_model_mle(
    states: np.ndarray,
    model_type: str = 'changepoint',
    save_results: bool = True,
) -> dict:
    """
    Fit Bayesian model using Maximum Likelihood Estimation.

    Parameters
    ----------
    states : np.ndarray
        Behavioral data array.
    model_type : str
        Model type: 'changepoint' or 'oddball'.
    save_results : bool
        Whether to save fitting results.

    Returns
    -------
    dict
        Fitting results including parameter estimates.
    """
    print(f"\nFitting {model_type} model using MLE...")

    # Create model
    model = BayesianModel(states, model_type=model_type)

    # Run MLE
    model.run_mle()

    # TODO: Extract results from model and return
    results = {
        'model_type': model_type,
        'method': 'MLE',
    }

    if save_results:
        output_dir = OUTPUT_DIR / 'bayesian_fits'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'mle_fit_{model_type}.npy'
        np.save(output_path, results)
        print(f"Saved results to: {output_path}")

    return results


def simulate_model(
    states: np.ndarray,
    model_type: str = 'changepoint',
    n_trials: int = 200,
    save_plots: bool = True,
) -> np.ndarray:
    """
    Simulate data from fitted Bayesian model.

    Parameters
    ----------
    states : np.ndarray
        Original behavioral data for fitting.
    model_type : str
        Model type: 'changepoint' or 'oddball'.
    n_trials : int
        Number of trials to simulate.
    save_plots : bool
        Whether to save visualization plots.

    Returns
    -------
    np.ndarray
        Simulated states array.
    """
    print(f"\nSimulating {n_trials} trials from {model_type} model...")

    # Create model
    model = BayesianModel(states, model_type=model_type)

    # Simulate data
    sim_states = model.sim_data(
        total_trials=n_trials,
        model_name="flexible_normative_model",
        condition=model_type,
    )

    if save_plots:
        plot_comparison(states, sim_states, model_type)

    return sim_states


def plot_comparison(
    real_states: np.ndarray,
    sim_states: np.ndarray,
    model_type: str,
):
    """
    Plot comparison between real and simulated data.

    Parameters
    ----------
    real_states : np.ndarray
        Real behavioral data.
    sim_states : np.ndarray
        Simulated data from model.
    model_type : str
        Model type for plot title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    contexts = ['Real Data', 'Simulated Data']
    states_list = [real_states, sim_states]

    for ax, context, states in zip(axes, contexts, states_list):
        trials = states[0]
        bucket_pos = states[1]
        bag_pos = states[2]
        heli_pos = states[3]

        ax.scatter(trials, bag_pos, label='Bag', color='red',
                   marker='o', alpha=0.7, edgecolors='k')
        ax.plot(trials, heli_pos, label='Helicopter', color='green', linewidth=3)
        ax.plot(trials, bucket_pos, label='Bucket', color='orange',
                alpha=0.8, linewidth=2)

        ax.set_ylim(-10, 310)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Position')
        ax.set_title(f'{context} ({model_type})')
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Bayesian Model Fit: {model_type.capitalize()}', fontsize=14)
    plt.tight_layout()

    # Save figure
    output_dir = BEHAVIORAL_FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'bayesian_fit_comparison_{model_type}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fit Bayesian normative models to behavioral data"
    )
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to behavioral data (.npy file)',
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='changepoint',
        choices=['changepoint', 'oddball'],
        help='Model type to fit',
    )
    parser.add_argument(
        '--method',
        type=str,
        default='mle',
        choices=['mle', 'simulate'],
        help='Fitting method: MLE or simulate',
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=200,
        help='Number of trials for simulation',
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results',
    )

    args = parser.parse_args()

    # Load data
    states = load_behavioral_data(args.data_path)

    # Fit or simulate
    if args.method == 'mle':
        results = fit_model_mle(
            states,
            model_type=args.model_type,
            save_results=not args.no_save,
        )
        print("\nFitting complete!")
    else:  # simulate
        sim_states = simulate_model(
            states,
            model_type=args.model_type,
            n_trials=args.n_trials,
            save_plots=not args.no_save,
        )
        print("\nSimulation complete!")


if __name__ == "__main__":
    main()
