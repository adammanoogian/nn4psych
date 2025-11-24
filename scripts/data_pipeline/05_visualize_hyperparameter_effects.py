#!/usr/bin/env python3
"""
Stage 05: Visualize Hyperparameter Effects

Analyzes and visualizes how hyperparameters affect model learning behavior:
- Gamma (discount factor) effects on learning rate area
- Rollout size effects
- Preset memory effects
- TD scale effects
- Combined parameter comparison plots

Input: Trained models with varied hyperparameters
Output: Figures in figures/parameter_exploration/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from torch.distributions import Categorical
import torch

from config import (
    TRAINED_MODELS_DIR, CHECKPOINTS_DIR, EXPLORATION_FIGURES_DIR,
    PARAMETER_EXPLORATION_DIR, MODEL_PARAMS, TASK_PARAMS,
    GAMMA_VALUES, ROLLOUT_VALUES, PRESET_VALUES, SCALE_VALUES
)
from nn4psych.models import ActorCritic
from envs import PIE_CP_OB_v2
from nn4psych.utils.metrics import get_lrs_v2
from nn4psych.utils.io import saveload


def extract_behavior_for_model(model_path, epochs=30, reset_memory=0.0):
    """Extract behavioral data from a single model."""
    params = MODEL_PARAMS['actor_critic']
    hidden_dim = params['hidden_dim']
    n_trials = 200

    model = ActorCritic(params['input_dim'], hidden_dim, params['action_dim'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    contexts = ["change-point", "oddball"]
    all_states = np.zeros([epochs, 2, 5, n_trials])

    with torch.no_grad():
        for epoch in range(epochs):
            for tt, context in enumerate(contexts):
                env = PIE_CP_OB_v2(
                    condition=context, total_trials=n_trials, max_time=300,
                    max_displacement=10, reward_size=5.0
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


def calculate_area_metric(states, threshold=20):
    """Calculate area under learning rate curve (CP - OB difference)."""
    epochs = states.shape[0]
    areas = {'cp': [], 'ob': []}

    for c, cond in enumerate(['cp', 'ob']):
        pes_all, lrs_all = [], []
        for e in range(epochs):
            pe, lr = get_lrs_v2(states[e, c], threshold=threshold)
            pes_all.extend(pe)
            lrs_all.extend(lr)

        if len(pes_all) > 0:
            pes = np.array(pes_all)
            lrs = np.array(lrs_all)
            sorted_idx = np.argsort(pes)
            area = np.trapezoid(lrs[sorted_idx], pes[sorted_idx])
            areas[cond] = area
        else:
            areas[cond] = 0

    return areas['cp'] - areas['ob']


def get_model_area(model_path, epochs=30):
    """Get the learning rate area difference for a model."""
    try:
        states = extract_behavior_for_model(str(model_path), epochs=epochs)
        return calculate_area_metric(states)
    except Exception as e:
        print(f"    Error: {e}")
        return None


def analyze_parameter(param_name, param_values, model_dir, epochs=8, max_models=5):
    """Analyze effect of a single parameter across values."""
    model_dir = Path(model_dir)
    results = {}

    print(f"\n  Analyzing {param_name}...")

    for val in param_values:
        # Build glob pattern based on parameter
        if param_name == 'gamma':
            pattern = f"*_V3_{val}g_0.0rm_100bz_*_1.0tds_*_64n_50000e_10md_5.0rz_*s.pth"
        elif param_name == 'rollout':
            pattern = f"*_V3_0.95g_0.0rm_{val}bz_*_1.0tds_*_64n_50000e_10md_5.0rz_*s.pth"
        elif param_name == 'preset':
            pattern = f"*_V3_0.95g_{val}rm_100bz_*_1.0tds_*_64n_50000e_10md_5.0rz_*s.pth"
        elif param_name == 'scale':
            pattern = f"*_V3_0.95g_0.0rm_100bz_*_{val}tds_*_64n_50000e_10md_5.0rz_*s.pth"
        else:
            continue

        files = list(model_dir.glob(pattern))[:max_models]

        if not files:
            print(f"    {param_name}={val}: No models found")
            continue

        areas = []
        for model_path in files:
            area = get_model_area(model_path, epochs=epochs)
            if area is not None:
                areas.append(area)

        if areas:
            results[val] = {
                'mean': np.mean(areas),
                'std': np.std(areas),
                'n': len(areas),
                'values': areas
            }
            print(f"    {param_name}={val}: {np.mean(areas):.3f} +/- {np.std(areas):.3f} (n={len(areas)})")

    return results


def plot_parameter_effect(param_name, results, output_dir, epochs):
    """Plot effect of a parameter on learning area."""
    output_dir = Path(output_dir)

    if not results:
        print(f"  No results for {param_name}")
        return

    vals = sorted(results.keys())
    means = [results[v]['mean'] for v in vals]
    stds = [results[v]['std'] for v in vals]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(vals, means, yerr=stds, marker='o', capsize=4, linewidth=2,
                markersize=8, color='steelblue')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Labels based on parameter
    labels = {
        'gamma': ('Discount Factor ($\\gamma$)', 'Effect of Discount Factor'),
        'rollout': ('Rollout Size', 'Effect of Rollout Size'),
        'preset': ('Memory Reset Probability', 'Effect of Memory Reset'),
        'scale': ('TD Scale ($\\beta_\\delta$)', 'Effect of TD Scale')
    }
    xlabel, title = labels.get(param_name, (param_name, f'Effect of {param_name}'))

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Area Difference (CP - OB)')
    ax.set_title(title)
    plt.tight_layout()

    plt.savefig(output_dir / f'{param_name}_area_{epochs}e.png', dpi=150)
    plt.savefig(output_dir / f'{param_name}_area_{epochs}e.svg')
    plt.close()

    # Save data
    saveload(output_dir / f'{param_name}_area.pickle', results, 'save')

    print(f"  Saved {param_name} plot to {output_dir}")


def plot_combined_parameters(all_results, output_dir, epochs):
    """Plot all parameter effects in a single figure."""
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    param_info = {
        'gamma': ('$\\gamma$', GAMMA_VALUES),
        'rollout': ('Rollout', ROLLOUT_VALUES),
        'preset': ('$p_{reset}$', PRESET_VALUES),
        'scale': ('$\\beta_\\delta$', SCALE_VALUES)
    }

    for idx, (param_name, (label, _)) in enumerate(param_info.items()):
        ax = axes[idx]
        results = all_results.get(param_name, {})

        if results:
            vals = sorted(results.keys())
            means = [results[v]['mean'] for v in vals]
            stds = [results[v]['std'] for v in vals]
            ax.errorbar(vals, means, yerr=stds, marker='o', capsize=3, linewidth=2)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel('Area (CP - OB)')
        ax.set_title(f'Effect of {label}')

    plt.tight_layout()
    plt.savefig(output_dir / f'all_params_area.png', dpi=150)
    plt.savefig(output_dir / f'all_params_area.svg')
    plt.close()

    print(f"  Saved combined plot to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 05: Visualize Hyperparameter Effects")
    parser.add_argument('--model_dir', type=str,
                       default=str(CHECKPOINTS_DIR / 'model_params_101000'),
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str,
                       default=str(EXPLORATION_FIGURES_DIR),
                       help='Output directory for figures')
    parser.add_argument('--epochs', type=int, default=8,
                       help='Epochs per model for behavior extraction')
    parser.add_argument('--params', type=str, nargs='+',
                       default=['gamma', 'rollout'],
                       choices=['gamma', 'rollout', 'preset', 'scale', 'all'],
                       help='Which parameters to analyze')
    parser.add_argument('--max_models', type=int, default=3,
                       help='Maximum models per parameter value')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STAGE 05: VISUALIZE HYPERPARAMETER EFFECTS")
    print("="*60)
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: {args.params}")

    # Determine which parameters to analyze
    if 'all' in args.params:
        params_to_analyze = ['gamma', 'rollout', 'preset', 'scale']
    else:
        params_to_analyze = args.params

    param_values = {
        'gamma': GAMMA_VALUES,
        'rollout': ROLLOUT_VALUES,
        'preset': PRESET_VALUES,
        'scale': SCALE_VALUES
    }

    all_results = {}

    for param_name in params_to_analyze:
        results = analyze_parameter(
            param_name,
            param_values[param_name],
            args.model_dir,
            epochs=args.epochs,
            max_models=args.max_models
        )
        all_results[param_name] = results
        plot_parameter_effect(param_name, results, output_dir, args.epochs)

    # Combined plot
    if len(all_results) > 1:
        print("\n  Creating combined plot...")
        plot_combined_parameters(all_results, output_dir, args.epochs)

    print("\n" + "="*60)
    print(f"DONE! Figures saved to: {output_dir}")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
