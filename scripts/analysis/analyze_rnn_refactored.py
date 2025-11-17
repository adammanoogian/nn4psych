#!/usr/bin/env python3
"""
Analyze RNN Model Behavior

This script loads a trained RNN model and analyzes its behavioral patterns,
learning rates, and performance across change-point and oddball conditions.

Refactored to use nn4psych package structure.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from copy import deepcopy
import glob

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from nn4psych package
from nn4psych.models import ActorCritic
from nn4psych.envs import PIE_CP_OB_v2
from nn4psych.utils.metrics import get_lrs_v2
from nn4psych.utils.io import saveload, load_model
from nn4psych.utils.plotting import plot_behavior
from nn4psych.analysis.behavior import extract_behavior
from config import (
    MODEL_PARAMS,
    TASK_PARAMS,
    TRAINING_PARAMS,
    GAMMA_VALUES,
    BEHAVIORAL_FIGURES_DIR,
)


def analyze_rnn_model(
    model_path: str,
    n_epochs: int = 100,
    reset_memory: bool = True,
    save_plots: bool = True,
):
    """
    Analyze a trained RNN model's behavior.

    Parameters
    ----------
    model_path : str
        Path to trained model weights.
    n_epochs : int
        Number of epochs to run.
    reset_memory : bool
        Whether to reset hidden state between epochs.
    save_plots : bool
        Whether to save generated plots.
    """
    print(f"Analyzing model: {model_path}")

    # Load model
    model = ActorCritic(
        input_dim=MODEL_PARAMS['actor_critic']['input_dim'],
        hidden_dim=MODEL_PARAMS['actor_critic']['hidden_dim'],
        action_dim=MODEL_PARAMS['actor_critic']['action_dim'],
    )

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # Create environments
    contexts = ["change-point", "oddball"]

    for context in contexts:
        print(f"\nAnalyzing {context} condition...")

        # Create environment
        task_params = TASK_PARAMS['change_point' if context == "change-point" else 'oddball']
        env = PIE_CP_OB_v2(**task_params)

        # Extract behavior
        states_list = extract_behavior(
            model,
            env,
            n_epochs=n_epochs,
            reset_memory=reset_memory,
        )

        # Analyze learning rates
        all_pes = []
        all_lrs = []

        for epoch_states in states_list:
            pe, lr = get_lrs_v2(epoch_states, threshold=20)
            valid_idx = (pe >= 0) & (lr >= 0)
            all_pes.extend(pe[valid_idx])
            all_lrs.extend(lr[valid_idx])

        # Plot results
        if save_plots and len(all_pes) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Plot 1: Learning rate vs PE
            ax = axes[0, 0]
            ax.scatter(all_pes, all_lrs, alpha=0.3, s=1)
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Learning Rate')
            ax.set_title(f'{context.capitalize()}: Learning Rate vs PE')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            # Plot 2: Binned learning rates
            ax = axes[0, 1]
            bins = np.linspace(20, 150, 20)
            pe_binned = pd.cut(all_pes, bins=bins)
            lr_by_bin = pd.Series(all_lrs).groupby(pe_binned).agg(['mean', 'std', 'count'])
            lr_by_bin = lr_by_bin[lr_by_bin['count'] >= 10]

            if len(lr_by_bin) > 0:
                x = [interval.mid for interval in lr_by_bin.index]
                y = lr_by_bin['mean'].values
                yerr = lr_by_bin['std'].values / np.sqrt(lr_by_bin['count'].values)

                ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=3)
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Mean Learning Rate')
            ax.set_title(f'{context.capitalize()}: Binned Learning Rates')
            ax.grid(True, alpha=0.3)

            # Plot 3: Sample trial behavior
            ax = axes[1, 0]
            sample_states = states_list[0]  # First epoch
            plot_behavior(sample_states, context, epoch=0, ax=ax)

            # Plot 4: Learning rate distribution
            ax = axes[1, 1]
            ax.hist(all_lrs, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{context.capitalize()}: LR Distribution')
            ax.axvline(np.mean(all_lrs), color='red', linestyle='--',
                      label=f'Mean: {np.mean(all_lrs):.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.suptitle(f'RNN Analysis: {Path(model_path).stem}', fontsize=14)
            plt.tight_layout()

            # Save
            output_dir = BEHAVIORAL_FIGURES_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"rnn_analysis_{context}_{Path(model_path).stem}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {output_path}")

        print(f"  Mean LR: {np.mean(all_lrs):.3f}")
        print(f"  Mean PE: {np.mean(all_pes):.3f}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze RNN model behavior")
    parser.add_argument('model_path', type=str, help='Path to model weights')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--no-reset', action='store_true', help='Do not reset memory between epochs')
    parser.add_argument('--no-save', action='store_true', help='Do not save plots')

    args = parser.parse_args()

    # Support glob patterns
    if '*' in args.model_path:
        files = glob.glob(args.model_path)
        if not files:
            print(f"No files found matching: {args.model_path}")
            return

        # Sort by performance (first number in filename)
        sorted_files = sorted(files, key=lambda x: float(Path(x).stem.split('_')[0]))
        print(f"Found {len(sorted_files)} models")

        # Analyze top 3
        for model_path in sorted_files[-3:]:
            analyze_rnn_model(
                model_path,
                n_epochs=args.epochs,
                reset_memory=not args.no_reset,
                save_plots=not args.no_save,
            )
    else:
        analyze_rnn_model(
            args.model_path,
            n_epochs=args.epochs,
            reset_memory=not args.no_reset,
            save_plots=not args.no_save,
        )


if __name__ == "__main__":
    # Need pandas for binning
    import pandas as pd
    main()
