#!/usr/bin/env python3
"""
Stage 04: Visualize Behavioral Summary

Generates behavioral analysis figures from trained models:
- State trajectory plots (helicopter, bucket, bag positions)
- Learning rate vs prediction error curves
- Learning rate histograms by condition
- Update by prediction error plots
- Interaction plots

Input: Trained model files (.pth) or extracted behavioral data
Output: Figures in figures/behavioral_summary/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from torch.distributions import Categorical
import torch

from config import (
    TRAINED_MODELS_DIR, CHECKPOINTS_DIR, BEHAVIORAL_FIGURES_DIR,
    MODEL_PARAMS, TASK_PARAMS
)
from nn4psych.models import ActorCritic
from envs import PIE_CP_OB_v2
from nn4psych.utils.metrics import get_lrs_v2


def extract_model_behavior(model_path, epochs=30, reset_memory=0.0):
    """Extract behavioral data from a trained model."""
    params = MODEL_PARAMS['actor_critic']
    hidden_dim = params['hidden_dim']
    n_trials = TASK_PARAMS['change_point']['total_trials']

    model = ActorCritic(
        params['input_dim'], hidden_dim, params['action_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    contexts = ["change-point", "oddball"]
    all_states = np.zeros([epochs, 2, 5, n_trials])

    with torch.no_grad():
        for epoch in range(epochs):
            for tt, context in enumerate(contexts):
                env = PIE_CP_OB_v2(
                    condition=context,
                    total_trials=n_trials,
                    max_time=TASK_PARAMS['change_point']['max_time'],
                    max_displacement=TASK_PARAMS['change_point']['max_displacement'],
                    reward_size=TASK_PARAMS['change_point']['reward_size'],
                )

                hx = torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5

                for trial in range(n_trials):
                    next_obs, done = env.reset()
                    norm_next_obs = env.normalize_states(next_obs)
                    next_state = np.concatenate([norm_next_obs, env.context, [0.0]])
                    next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

                    while not done:
                        if np.random.random_sample() < reset_memory:
                            hx = torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5

                        actor_logits, _, hx = model(next_state, hx)
                        probs = Categorical(logits=actor_logits)
                        action = probs.sample()

                        next_obs, reward, done = env.step(action.item())
                        norm_next_obs = env.normalize_states(next_obs)
                        next_state = np.concatenate([norm_next_obs, env.context, [reward]])
                        next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

                all_states[epoch, tt] = np.array([
                    env.trials, env.bucket_positions, env.bag_positions,
                    env.helicopter_positions, env.hazard_triggers
                ])

    return all_states


def get_learning_rates(states, threshold=20):
    """Extract and aggregate learning rates across epochs."""
    epochs = states.shape[0]
    results = {'cp': {'pes': [], 'lrs': []}, 'ob': {'pes': [], 'lrs': []}}

    for c, cond in enumerate(['cp', 'ob']):
        for e in range(epochs):
            pe, lr = get_lrs_v2(states[e, c], threshold=threshold)
            results[cond]['pes'].extend(pe)
            results[cond]['lrs'].extend(lr)

        # Sort by prediction error
        pes = np.array(results[cond]['pes'])
        lrs = np.array(results[cond]['lrs'])
        sorted_idx = np.argsort(pes)
        results[cond]['pes'] = pes[sorted_idx]
        results[cond]['lrs'] = lrs[sorted_idx]
        results[cond]['area'] = np.trapezoid(lrs[sorted_idx], pes[sorted_idx])

    return results


def plot_state_trajectories(states, output_dir, epoch=0, params=None):
    """Plot trial-by-trial state trajectories for each condition."""
    output_dir = Path(output_dir)
    contexts = ["Change-point", "Oddball"]
    filenames = ["Helicopter_CP", "Helicopter_OB"]

    for c, (context, fname) in enumerate(zip(contexts, filenames)):
        trials, bucket, bag, heli, hazards = states[epoch, c]

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.scatter(trials, bag, label='Bag', color='red', s=20, alpha=0.7, edgecolors='k')
        ax.plot(trials, heli, label='Helicopter', color='green', linewidth=2)
        ax.plot(trials, bucket, label='Bucket', color='orange', linewidth=2)

        # Mark hazard trials
        hazard_idx = np.where(np.array(hazards) == 1)[0]
        if len(hazard_idx) > 0:
            ax.scatter(np.array(trials)[hazard_idx], np.array(bag)[hazard_idx],
                      color='purple', s=50, marker='*', label='Hazard', zorder=5)

        ax.set_ylim(-10, 310)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Position')
        ax.set_title(f'{context} Condition')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()

        plt.savefig(output_dir / f'{fname}.png', dpi=150)
        plt.savefig(output_dir / f'{fname}.svg')
        plt.close()

    print(f"  Saved state trajectory plots to {output_dir}")


def plot_learning_rate_curves(results, output_dir, scale=0.05):
    """Plot learning rate vs prediction error curves."""
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(4, 3))
    colors = {'cp': 'orange', 'ob': 'brown'}
    labels = {'cp': 'Change-point', 'ob': 'Oddball'}

    for cond in ['cp', 'ob']:
        pes = results[cond]['pes']
        lrs = results[cond]['lrs']
        if len(lrs) > 0:
            window = max(1, int(len(lrs) * scale))
            smoothed = uniform_filter1d(lrs, size=window)
            ax.plot(pes, smoothed, color=colors[cond], linewidth=2, label=labels[cond])

    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate by Prediction Error')
    ax.legend()
    ax.set_xlim(20, 150)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    plt.savefig(output_dir / 'learning_rate_by_prediction_error.png', dpi=150)
    plt.savefig(output_dir / 'learning_rate_vs_prediction_error.png', dpi=150)
    plt.close()

    print(f"  Saved learning rate curves to {output_dir}")


def plot_learning_rate_histograms(results, output_dir):
    """Plot learning rate histograms for each condition."""
    output_dir = Path(output_dir)
    conditions = {'cp': 'change-point', 'ob': 'oddball'}

    for cond, name in conditions.items():
        lrs = results[cond]['lrs']
        if len(lrs) == 0:
            continue

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(lrs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Count')
        ax.set_title(f'Learning Rate Distribution ({name.title()})')
        ax.set_xlim(0, 1)
        plt.tight_layout()

        plt.savefig(output_dir / f'learning_rate_histogram_{name}.png', dpi=150)
        plt.close()

    # Combined histogram
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(results['cp']['lrs'], bins=30, alpha=0.6, label='CP', color='orange')
    ax.hist(results['ob']['lrs'], bins=30, alpha=0.6, label='OB', color='brown')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate_histogram.png', dpi=150)
    plt.close()

    print(f"  Saved learning rate histograms to {output_dir}")


def plot_update_by_pe(results, output_dir):
    """Plot update magnitude by prediction error."""
    output_dir = Path(output_dir)
    conditions = {'cp': 'change-point', 'ob': 'oddball'}

    for cond, name in conditions.items():
        pes = results[cond]['pes']
        lrs = results[cond]['lrs']
        if len(pes) == 0:
            continue

        # Update = LR * PE
        updates = lrs * pes

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(pes, updates, alpha=0.3, s=10)
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Update')
        ax.set_title(f'Update by Prediction Error ({name.title()})')
        plt.tight_layout()

        plt.savefig(output_dir / f'update_by_prediction_error_{name}.png', dpi=150)
        plt.close()

    print(f"  Saved update plots to {output_dir}")


def plot_states_and_lr_over_trials(states, output_dir, epoch=0):
    """Plot states and learning rate over trials."""
    output_dir = Path(output_dir)
    conditions = {0: 'change-point', 1: 'oddball'}

    for c, name in conditions.items():
        trials, bucket, bag, heli, hazards = states[epoch, c]

        # Calculate per-trial learning rate
        pe = np.array(bag[:-1]) - np.array(bucket[:-1])
        update = np.diff(bucket)
        lr = np.where(pe != 0, update / pe, 0)
        lr = np.clip(lr, 0, 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

        # Top: positions
        ax1.plot(trials, heli, label='Helicopter', color='green', linewidth=1.5)
        ax1.plot(trials, bucket, label='Bucket', color='orange', linewidth=1.5)
        ax1.scatter(trials, bag, label='Bag', color='red', s=10, alpha=0.5)
        ax1.set_ylabel('Position')
        ax1.legend(loc='upper right', fontsize=7)
        ax1.set_title(f'{name.title()} Condition')

        # Bottom: learning rate
        ax2.plot(trials[:-1], lr, color='purple', linewidth=1)
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Learning Rate')
        ax2.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(output_dir / f'states_and_learning_rate_over_trials_{name}.png', dpi=150)
        plt.close()

    print(f"  Saved trial-by-trial plots to {output_dir}")


def find_best_model(model_dir):
    """Find best performing model in directory."""
    model_dir = Path(model_dir)
    files = list(model_dir.glob("*_V3_0.95g_0.0rm_100bz_*_1.0tds_*_64n_50000e_*.pth"))

    if not files:
        files = list(model_dir.glob("*.pth"))

    if not files:
        return None

    # Sort by loss (first number in filename)
    try:
        sorted_files = sorted(files, key=lambda x: float(x.name.split('_')[0]))
        return sorted_files[0]
    except:
        return files[0]


def main():
    parser = argparse.ArgumentParser(description="Stage 04: Visualize Behavioral Summary")
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to specific model (auto-selects best if not provided)')
    parser.add_argument('--model_dir', type=str,
                       default=str(CHECKPOINTS_DIR / 'model_params_101000'),
                       help='Directory containing models')
    parser.add_argument('--output_dir', type=str,
                       default=str(BEHAVIORAL_FIGURES_DIR),
                       help='Output directory for figures')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs for behavior extraction')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STAGE 04: VISUALIZE BEHAVIORAL SUMMARY")
    print("="*60)

    # Find model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        print("\n1. Finding best model...")
        model_path = find_best_model(args.model_dir)

    if model_path is None:
        print("ERROR: No model found!")
        return 1

    print(f"   Using: {model_path.name}")

    # Extract behavior
    print(f"\n2. Extracting behavior ({args.epochs} epochs)...")
    states = extract_model_behavior(str(model_path), epochs=args.epochs)
    print(f"   Shape: {states.shape}")

    # Calculate learning rates
    print("\n3. Computing learning rates...")
    results = get_learning_rates(states)
    print(f"   CP area: {results['cp']['area']:.2f}")
    print(f"   OB area: {results['ob']['area']:.2f}")

    # Generate all figures
    print("\n4. Generating figures...")
    plot_state_trajectories(states, output_dir)
    plot_learning_rate_curves(results, output_dir)
    plot_learning_rate_histograms(results, output_dir)
    plot_update_by_pe(results, output_dir)
    plot_states_and_lr_over_trials(states, output_dir)

    print("\n" + "="*60)
    print(f"DONE! Figures saved to: {output_dir}")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
