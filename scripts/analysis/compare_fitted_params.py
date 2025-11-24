#!/usr/bin/env python3
"""
Compare Our Fitted Parameters to Nassar et al. 2021 Reference Parameters

This script compares the Bayesian parameters fitted by our NumPyro implementation
to the reference parameters from Nassar et al. (2021) to validate our fitting procedure.

We expect high correlation if our implementation correctly reproduces their model.

Usage:
    python compare_fitted_params.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import OUTPUT_DIR, BEHAVIORAL_FIGURES_DIR

# Nassar parameter file
NASSAR_PARAM_FILE = Path('C:/Users/aman0087/Documents/Github/Nassar_et_al_2021/Brain2021Code/heliParamEstimatesForJim_23-Nov-2021.mat')


def load_nassar_reference_params():
    """
    Load Nassar's fitted parameters from MATLAB file.

    Returns
    -------
    pd.DataFrame
        Reference parameters for all subjects
    """
    print(f"Loading Nassar reference parameters from: {NASSAR_PARAM_FILE}")

    data = sio.loadmat(str(NASSAR_PARAM_FILE))
    sub_data = data['subData']

    # Extract parameter labels
    param_labels = sub_data['paramLabels'][0][0][0]
    param_labels = [label[0] for label in param_labels]

    print(f"\nNassar parameter labels: {param_labels}")
    print(f"  [0] HAZ - Hazard rate (our H)")
    print(f"  [1] LW - Likelihood weight (our LW)")
    print(f"  [2] UD - Uncertainty discount (our UU)")
    print(f"  [3] UP STD - Update std (our sigma_motor)")
    print(f"  [4] UP STDslope - Update std slope (our sigma_LR)")

    # Extract data
    params = sub_data['params'][0][0]  # Shape: (134, 10)
    sub_names = sub_data['subName'][0][0]
    is_patient = sub_data['isPatient'][0][0]

    n_subjects = len(params)
    print(f"\nTotal subjects in reference file: {n_subjects}")

    # Build dataframe
    rows = []
    for i in range(n_subjects):
        subject_id = sub_names[i][0][0] if isinstance(sub_names[i][0], np.ndarray) else sub_names[i]
        patient_status = bool(is_patient[i][0][0] if isinstance(is_patient[i][0], np.ndarray) else is_patient[i])
        param_values = params[i].flatten()

        rows.append({
            'subject_id': subject_id,
            'is_patient': patient_status,
            'nassar_H': param_values[0],         # HAZ
            'nassar_LW': param_values[1],        # LW
            'nassar_UU': param_values[2],        # UD
            'nassar_sigma_motor': param_values[3],  # UP STD
            'nassar_sigma_LR': param_values[4],     # UP STDslope
        })

    ref_df = pd.DataFrame(rows)

    print(f"\nReference parameter ranges:")
    print(ref_df[['nassar_H', 'nassar_LW', 'nassar_UU', 'nassar_sigma_motor', 'nassar_sigma_LR']].describe())

    return ref_df


def load_our_fitted_params():
    """
    Load our NumPyro fitted parameters.

    Returns
    -------
    pd.DataFrame
        Our fitted parameters
    """
    params_path = OUTPUT_DIR / 'fitted_params' / 'nassar2021' / 'numpyro_fitted_params.csv'

    if not params_path.exists():
        raise FileNotFoundError(
            f"Fitted parameters not found at {params_path}\n"
            f"Run scripts/fitting/fit_nassar_numpyro.py first"
        )

    print(f"\nLoading our fitted parameters from: {params_path}")
    our_df = pd.DataFrame(pd.read_csv(params_path))

    print(f"Total fits: {len(our_df)}")
    print(f"  Changepoint: {(our_df['context'] == 'changepoint').sum()}")
    print(f"  Oddball: {(our_df['context'] == 'oddball').sum()}")

    return our_df


def compare_parameters(our_df, ref_df, context='changepoint'):
    """
    Compare our fitted parameters to Nassar's for a specific context.

    Parameters
    ----------
    our_df : pd.DataFrame
        Our fitted parameters
    ref_df : pd.DataFrame
        Nassar's reference parameters
    context : str
        'changepoint' or 'oddball'

    Returns
    -------
    pd.DataFrame
        Merged comparison dataframe
    """
    print(f"\n{'='*70}")
    print(f"COMPARING PARAMETERS: {context.upper()} CONTEXT")
    print(f"{'='*70}")

    # Filter our data by context
    our_context = our_df[our_df['context'] == context].copy()

    # Merge with reference parameters
    merged = our_context.merge(ref_df, on='subject_id', suffixes=('', '_ref'))

    print(f"\nMatched subjects: {len(merged)}")

    # Compute correlations
    params = ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']

    print(f"\n[CORRELATION ANALYSIS]")
    print(f"{'Parameter':<15} {'Pearson r':<12} {'p-value':<12} {'Spearman ρ':<12}")
    print("-" * 60)

    correlations = {}

    for param in params:
        our_vals = merged[param].values
        ref_vals = merged[f'nassar_{param}'].values

        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(our_vals, ref_vals)

        # Spearman correlation (robust to outliers)
        r_spearman, p_spearman = stats.spearmanr(our_vals, ref_vals)

        print(f"{param:<15} {r_pearson:>11.3f} {p_pearson:>11.4f} {r_spearman:>11.3f}")

        correlations[param] = {
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman,
        }

    # Mean absolute error
    print(f"\n[MEAN ABSOLUTE ERROR]")
    print(f"{'Parameter':<15} {'MAE':<12} {'RMSE':<12}")
    print("-" * 40)

    for param in params:
        our_vals = merged[param].values
        ref_vals = merged[f'nassar_{param}'].values

        mae = np.mean(np.abs(our_vals - ref_vals))
        rmse = np.sqrt(np.mean((our_vals - ref_vals)**2))

        print(f"{param:<15} {mae:>11.3f} {rmse:>11.3f}")

        correlations[param]['mae'] = mae
        correlations[param]['rmse'] = rmse

    return merged, correlations


def plot_parameter_comparison(merged, context='changepoint', save_path=None):
    """
    Create scatter plots comparing our parameters to Nassar's.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged comparison data
    context : str
        Context name for title
    save_path : Path, optional
        Path to save figure
    """
    params = ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']
    param_labels = [
        'Hazard Rate (H)',
        'Likelihood Weight (LW)',
        'Uncertainty Underest. (UU)',
        'Motor Variance (σ_motor)',
        'LR Variance Slope (σ_LR)'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i]

        our_vals = merged[param].values
        ref_vals = merged[f'nassar_{param}'].values

        # Scatter plot with patient/control coloring
        patients = merged['is_patient_x'].values
        colors = ['red' if p else 'blue' for p in patients]

        ax.scatter(ref_vals, our_vals, c=colors, alpha=0.5, s=40)

        # Identity line
        min_val = min(ref_vals.min(), our_vals.min())
        max_val = max(ref_vals.max(), our_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

        # Correlation
        r, p = stats.pearsonr(our_vals, ref_vals)

        ax.set_xlabel(f'Nassar {label}', fontsize=10)
        ax.set_ylabel(f'Our Fitted {label}', fontsize=10)
        ax.set_title(f'r = {r:.3f}, p = {p:.4f}', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.5, label='Patients'),
        Patch(facecolor='blue', alpha=0.5, label='Controls')
    ]
    axes[5].legend(handles=legend_elements, loc='center', fontsize=12)
    axes[5].axis('off')

    plt.suptitle(f'Parameter Comparison: {context.capitalize()} Context\nOur NumPyro Fits vs Nassar Reference',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison plot: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_patient_control_comparison(our_df, ref_df):
    """
    Compare patient vs control parameter distributions.

    Parameters
    ----------
    our_df : pd.DataFrame
        Our fitted parameters
    ref_df : pd.DataFrame
        Nassar reference parameters
    """
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    params = ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']
    param_labels = ['H', 'LW', 'UU', 'σ_motor', 'σ_LR']

    # Top row: Our fits (changepoint)
    our_cp = our_df[our_df['context'] == 'changepoint']

    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[0, i]

        patients = our_cp[our_cp['is_patient']][param]
        controls = our_cp[~our_cp['is_patient']][param]

        ax.hist(patients, bins=15, alpha=0.6, color='red', label='Patients', density=True)
        ax.hist(controls, bins=15, alpha=0.6, color='blue', label='Controls', density=True)

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Our Fits - {label}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Bottom row: Nassar reference
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[1, i]

        patients = ref_df[ref_df['is_patient']][f'nassar_{param}']
        controls = ref_df[~ref_df['is_patient']][f'nassar_{param}']

        ax.hist(patients, bins=15, alpha=0.6, color='red', label='Patients', density=True)
        ax.hist(controls, bins=15, alpha=0.6, color='blue', label='Controls', density=True)

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Nassar Reference - {label}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Patient vs Control Parameter Distributions\nOur Fits (top) vs Nassar Reference (bottom)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = BEHAVIORAL_FIGURES_DIR / 'param_comparison_distributions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved distribution comparison: {save_path}")
    plt.close()


def main():
    """Main entry point."""
    print("=" * 70)
    print("COMPARING FITTED PARAMETERS TO NASSAR REFERENCE")
    print("=" * 70)

    # Load data
    ref_df = load_nassar_reference_params()
    our_df = load_our_fitted_params()

    # Compare for both contexts
    figures_dir = BEHAVIORAL_FIGURES_DIR / 'param_comparison'
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for context in ['changepoint', 'oddball']:
        merged, correlations = compare_parameters(our_df, ref_df, context=context)

        # Save comparison data
        output_path = OUTPUT_DIR / 'validation' / 'nassar2021' / f'param_comparison_{context}.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"\nSaved comparison data: {output_path}")

        # Plot comparison
        plot_path = figures_dir / f'param_comparison_{context}.png'
        plot_parameter_comparison(merged, context=context, save_path=plot_path)

        results[context] = {
            'merged': merged,
            'correlations': correlations,
        }

    # Plot patient/control distributions
    plot_patient_control_comparison(our_df, ref_df)

    # Overall validation summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")

    for context in ['changepoint', 'oddball']:
        print(f"\n{context.capitalize()}:")
        correlations = results[context]['correlations']

        # Average correlation
        avg_r = np.mean([correlations[p]['pearson_r'] for p in correlations])
        print(f"  Average Pearson correlation: {avg_r:.3f}")

        # Check if all significant
        all_sig = all(correlations[p]['pearson_p'] < 0.05 for p in correlations)
        print(f"  All parameters significantly correlated: {all_sig}")


if __name__ == "__main__":
    main()
