#!/usr/bin/env python3
"""
Validate Bayesian Model Fitting Against Nassar et al. 2021

This script reproduces key findings from Nassar et al. (2021) "All or Nothing"
to validate our Bayesian model fitting implementation before applying it to RNN agents.

Key findings to reproduce:
1. Learning rate curves for CP vs OB conditions
2. Area between curves (context discrimination metric)
3. Patient vs control differences
4. Model fits to human data

Reference:
    Nassar et al. (2021). All or nothing belief updating in patients with
    schizophrenia reduces precision and flexibility of beliefs.
"""

import sys
from pathlib import Path
import argparse
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import OUTPUT_DIR, BEHAVIORAL_FIGURES_DIR


class NassarDataLoader:
    """Load and parse Nassar et al. 2021 data."""

    def __init__(self, data_dir: str = 'data/raw/nassar2021'):
        """
        Initialize data loader.

        Parameters
        ----------
        data_dir : str
            Directory containing Nassar .mat files
        """
        self.data_dir = Path(data_dir)
        self.n_patients = 102  # First 102 of 134 are patients
        self.n_bins = 115      # Number of bins in sliding window analysis

    def load_data(self):
        """
        Load subject and model data from .mat files.

        Returns
        -------
        dict
            Dictionary with 'subjects' and 'model' data arrays
        """
        print("Loading Nassar et al. 2021 data...")

        # Load MATLAB files
        model_path = self.data_dir / 'slidingWindowFits_model_23-Nov-2021.mat'
        subject_path = self.data_dir / 'slidingWindowFits_subjects_23-Nov-2021.mat'

        if not model_path.exists() or not subject_path.exists():
            raise FileNotFoundError(
                f"Nassar data files not found in {self.data_dir}\n"
                f"Expected:\n"
                f"  - {model_path}\n"
                f"  - {subject_path}"
            )

        model_data = sio.loadmat(str(model_path))
        subject_data = sio.loadmat(str(subject_path))

        # Extract binRegData arrays
        # Structure: data[0][0][condition][participant][:][column]
        # condition: 0=CP (changepoint), 1=OB (oddball)
        # column 0: bin centers (relative error)
        # column 1: learning rate values
        model_data = np.asarray(model_data['binRegData'])
        subject_data = np.asarray(subject_data['binRegData'])

        print(f"  Loaded model data: {model_path.name}")
        print(f"  Loaded subject data: {subject_path.name}")
        print(f"  Number of participants: {len(subject_data[0][0][0])}")
        print(f"  Number of bins: {self.n_bins}")
        print(f"  Patients: {self.n_patients}")
        print(f"  Controls: {len(subject_data[0][0][0]) - self.n_patients}")

        return {
            'subjects': subject_data,
            'model': model_data
        }


class NassarAnalysis:
    """Analyze Nassar et al. 2021 data and reproduce key findings."""

    def __init__(self, data_dict: dict, n_patients: int = 102):
        """
        Initialize analysis.

        Parameters
        ----------
        data_dict : dict
            Dictionary from NassarDataLoader.load_data()
        n_patients : int
            Number of patients (rest are controls)
        """
        self.data = data_dict
        self.n_patients = n_patients
        self.cp_idx = 0  # Changepoint condition index
        self.ob_idx = 1  # Oddball condition index

    def extract_learning_rates(self, data_type: str = 'subjects'):
        """
        Extract learning rate curves for all groups.

        Parameters
        ----------
        data_type : str
            'subjects' or 'model'

        Returns
        -------
        dict
            Learning rate curves for each group/condition
        """
        data = self.data[data_type]

        # Extract data for each group and condition
        # Structure: data[0][0][condition][participant][:][column]
        # Column 1 contains learning rates

        results = {}

        # Patients
        cp_pat = data[0][0][self.cp_idx][:self.n_patients, :, 1]
        ob_pat = data[0][0][self.ob_idx][:self.n_patients, :, 1]

        results['patients_cp'] = {
            'mean': np.mean(cp_pat, axis=0),
            'std': np.std(cp_pat, axis=0),
            'sem': np.std(cp_pat, axis=0) / np.sqrt(self.n_patients),
            'individual': cp_pat
        }

        results['patients_ob'] = {
            'mean': np.mean(ob_pat, axis=0),
            'std': np.std(ob_pat, axis=0),
            'sem': np.std(ob_pat, axis=0) / np.sqrt(self.n_patients),
            'individual': ob_pat
        }

        # Controls
        n_controls = len(data[0][0][self.cp_idx]) - self.n_patients
        cp_ctrl = data[0][0][self.cp_idx][self.n_patients:, :, 1]
        ob_ctrl = data[0][0][self.ob_idx][self.n_patients:, :, 1]

        results['controls_cp'] = {
            'mean': np.mean(cp_ctrl, axis=0),
            'std': np.std(cp_ctrl, axis=0),
            'sem': np.std(cp_ctrl, axis=0) / np.sqrt(n_controls),
            'individual': cp_ctrl
        }

        results['controls_ob'] = {
            'mean': np.mean(ob_ctrl, axis=0),
            'std': np.std(ob_ctrl, axis=0),
            'sem': np.std(ob_ctrl, axis=0) / np.sqrt(n_controls),
            'individual': ob_ctrl
        }

        return results

    def compute_area_between_curves(self, lr_data: dict):
        """
        Compute area between CP and OB learning rate curves.

        This is the key metric for context discrimination.

        Parameters
        ----------
        lr_data : dict
            Learning rate data from extract_learning_rates()

        Returns
        -------
        dict
            Area between curves for patients and controls
        """
        # Calculate absolute difference between CP and OB curves
        area_patients = np.trapezoid(
            np.abs(lr_data['patients_cp']['mean'] - lr_data['patients_ob']['mean']),
            dx=1
        )

        area_controls = np.trapezoid(
            np.abs(lr_data['controls_cp']['mean'] - lr_data['controls_ob']['mean']),
            dx=1
        )

        return {
            'patients': area_patients,
            'controls': area_controls,
            'difference': area_controls - area_patients,
            'ratio': area_controls / area_patients if area_patients > 0 else np.inf
        }

    def statistical_comparison(self, lr_data: dict):
        """
        Perform statistical tests comparing groups.

        Parameters
        ----------
        lr_data : dict
            Learning rate data from extract_learning_rates()

        Returns
        -------
        dict
            Statistical test results
        """
        results = {}

        # Compare CP learning rates: patients vs controls
        # Use mean across bins for each participant
        pat_cp_means = np.mean(lr_data['patients_cp']['individual'], axis=1)
        ctrl_cp_means = np.mean(lr_data['controls_cp']['individual'], axis=1)

        t_stat_cp, p_val_cp = ttest_ind(pat_cp_means, ctrl_cp_means)
        results['cp_comparison'] = {
            't_statistic': float(t_stat_cp),
            'p_value': float(p_val_cp),
            'patients_mean': float(pat_cp_means.mean()),
            'controls_mean': float(ctrl_cp_means.mean())
        }

        # Compare OB learning rates: patients vs controls
        pat_ob_means = np.mean(lr_data['patients_ob']['individual'], axis=1)
        ctrl_ob_means = np.mean(lr_data['controls_ob']['individual'], axis=1)

        t_stat_ob, p_val_ob = ttest_ind(pat_ob_means, ctrl_ob_means)
        results['ob_comparison'] = {
            't_statistic': float(t_stat_ob),
            'p_value': float(p_val_ob),
            'patients_mean': float(pat_ob_means.mean()),
            'controls_mean': float(ctrl_ob_means.mean())
        }

        # Compare context discrimination (area between curves) at individual level
        # For each participant, compute their individual area
        pat_areas = []
        for i in range(len(lr_data['patients_cp']['individual'])):
            area = np.trapezoid(
                np.abs(lr_data['patients_cp']['individual'][i] -
                      lr_data['patients_ob']['individual'][i]),
                dx=1
            )
            pat_areas.append(area)

        ctrl_areas = []
        for i in range(len(lr_data['controls_cp']['individual'])):
            area = np.trapezoid(
                np.abs(lr_data['controls_cp']['individual'][i] -
                      lr_data['controls_ob']['individual'][i]),
                dx=1
            )
            ctrl_areas.append(area)

        t_stat_area, p_val_area = ttest_ind(pat_areas, ctrl_areas)
        results['area_comparison'] = {
            't_statistic': float(t_stat_area),
            'p_value': float(p_val_area),
            'patients_mean': float(np.mean(pat_areas)),
            'patients_std': float(np.std(pat_areas)),
            'controls_mean': float(np.mean(ctrl_areas)),
            'controls_std': float(np.std(ctrl_areas))
        }

        return results


def plot_figure_6a(lr_data: dict, save_path: Path = None):
    """
    Reproduce Nassar et al. 2021 Figure 6a (human subjects).

    Parameters
    ----------
    lr_data : dict
        Learning rate data for subjects
    save_path : Path, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_bins = len(lr_data['patients_cp']['mean'])
    x_axis = np.arange(n_bins)

    # Plot learning rate curves
    ax.plot(x_axis, lr_data['patients_cp']['mean'],
           label='Patients CP', color='orange', linewidth=2)
    ax.plot(x_axis, lr_data['patients_ob']['mean'],
           label='Patients OB', color='brown', linewidth=2)
    ax.plot(x_axis, lr_data['controls_cp']['mean'],
           label='Controls CP', color='blue', linewidth=2)
    ax.plot(x_axis, lr_data['controls_ob']['mean'],
           label='Controls OB', color='green', linewidth=2)

    # Add error bars for controls (as in original paper)
    ax.fill_between(x_axis,
                    lr_data['controls_cp']['mean'] - lr_data['controls_cp']['sem'],
                    lr_data['controls_cp']['mean'] + lr_data['controls_cp']['sem'],
                    color='blue', alpha=0.2)
    ax.fill_between(x_axis,
                    lr_data['controls_ob']['mean'] - lr_data['controls_ob']['sem'],
                    lr_data['controls_ob']['mean'] + lr_data['controls_ob']['sem'],
                    color='green', alpha=0.2)

    ax.set_xlabel('Relative Error (Bin)', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Human Subjects: Learning Rates by Condition and Group\n(Nassar et al. 2021, Figure 6a)',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 6a: {save_path}")

    return fig


def plot_figure_6b(lr_data: dict, save_path: Path = None):
    """
    Reproduce Nassar et al. 2021 Figure 6b (model fits).

    Parameters
    ----------
    lr_data : dict
        Learning rate data for model
    save_path : Path, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_bins = len(lr_data['patients_cp']['mean'])
    x_axis = np.arange(n_bins)

    # Plot model fit curves
    ax.plot(x_axis, lr_data['patients_cp']['mean'],
           label='Model: Patients CP', color='orange', linewidth=2, linestyle='--')
    ax.plot(x_axis, lr_data['patients_ob']['mean'],
           label='Model: Patients OB', color='brown', linewidth=2, linestyle='--')
    ax.plot(x_axis, lr_data['controls_cp']['mean'],
           label='Model: Controls CP', color='blue', linewidth=2, linestyle='--')
    ax.plot(x_axis, lr_data['controls_ob']['mean'],
           label='Model: Controls OB', color='green', linewidth=2, linestyle='--')

    ax.set_xlabel('Relative Error (Bin)', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Bayesian Model Fits: Learning Rates by Condition and Group\n(Nassar et al. 2021, Figure 6b)',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 6b: {save_path}")

    return fig


def plot_area_comparison(area_subjects: dict, area_model: dict, save_path: Path = None):
    """
    Plot bar graph comparing area between curves.

    Parameters
    ----------
    area_subjects : dict
        Area results for subjects
    area_model : dict
        Area results for model
    save_path : Path, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Subjects\nPatients', 'Subjects\nControls',
                 'Model\nPatients', 'Model\nControls']
    values = [area_subjects['patients'], area_subjects['controls'],
             area_model['patients'], area_model['controls']]
    colors = ['orange', 'blue', 'coral', 'lightblue']

    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Area Between CP and OB Curves', fontsize=12)
    ax.set_title('Context Discrimination: Area Between Curves\n(Higher = Better Discrimination)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved area comparison: {save_path}")

    return fig


def print_validation_summary(area_subjects: dict, area_model: dict, stats: dict):
    """
    Print comprehensive validation summary.

    Parameters
    ----------
    area_subjects : dict
        Area between curves for subjects
    area_model : dict
        Area between curves for model
    stats : dict
        Statistical test results
    """
    print("\n" + "="*70)
    print("NASSAR ET AL. 2021 VALIDATION SUMMARY")
    print("="*70)

    print("\n[DATA] AREA BETWEEN CURVES (Context Discrimination)")
    print("-" * 70)
    print(f"  Subjects - Patients: {area_subjects['patients']:.3f}")
    print(f"  Subjects - Controls: {area_subjects['controls']:.3f}")
    print(f"  Difference (C - P):  {area_subjects['difference']:.3f}")
    print(f"  Ratio (C / P):       {area_subjects['ratio']:.3f}x")
    print()
    print(f"  Model - Patients:    {area_model['patients']:.3f}")
    print(f"  Model - Controls:    {area_model['controls']:.3f}")
    print(f"  Difference (C - P):  {area_model['difference']:.3f}")
    print(f"  Ratio (C / P):       {area_model['ratio']:.3f}x")

    print("\n[STATS] STATISTICAL COMPARISONS")
    print("-" * 70)
    print("  Changepoint Condition (Patients vs Controls):")
    print(f"    Patients mean LR:  {stats['cp_comparison']['patients_mean']:.3f}")
    print(f"    Controls mean LR:  {stats['cp_comparison']['controls_mean']:.3f}")
    print(f"    t-statistic:       {stats['cp_comparison']['t_statistic']:.3f}")
    print(f"    p-value:           {stats['cp_comparison']['p_value']:.4f}")

    print("\n  Oddball Condition (Patients vs Controls):")
    print(f"    Patients mean LR:  {stats['ob_comparison']['patients_mean']:.3f}")
    print(f"    Controls mean LR:  {stats['ob_comparison']['controls_mean']:.3f}")
    print(f"    t-statistic:       {stats['ob_comparison']['t_statistic']:.3f}")
    print(f"    p-value:           {stats['ob_comparison']['p_value']:.4f}")

    print("\n  Context Discrimination (Area Between Curves):")
    print(f"    Patients mean:     {stats['area_comparison']['patients_mean']:.3f} ± {stats['area_comparison']['patients_std']:.3f}")
    print(f"    Controls mean:     {stats['area_comparison']['controls_mean']:.3f} ± {stats['area_comparison']['controls_std']:.3f}")
    print(f"    t-statistic:       {stats['area_comparison']['t_statistic']:.3f}")
    print(f"    p-value:           {stats['area_comparison']['p_value']:.4f}")

    print("\n[CHECKS] VALIDATION CHECKS")
    print("-" * 70)

    # Check 1: Controls should have larger area than patients
    check1 = area_subjects['controls'] > area_subjects['patients']
    print(f"  [{'PASS' if check1 else 'FAIL'}] Controls have larger context discrimination than patients")

    # Check 2: Model should capture this pattern
    check2 = area_model['controls'] > area_model['patients']
    print(f"  [{'PASS' if check2 else 'FAIL'}] Model captures patient/control difference")

    # Check 3: Area difference is statistically significant
    check3 = stats['area_comparison']['p_value'] < 0.05
    print(f"  [{'PASS' if check3 else 'FAIL'}] Area difference is statistically significant (p < 0.05)")

    # Check 4: Learning rates are in valid range [0, 1]
    check4 = (0 <= stats['cp_comparison']['patients_mean'] <= 1 and
             0 <= stats['cp_comparison']['controls_mean'] <= 1)
    print(f"  [{'PASS' if check4 else 'FAIL'}] Learning rates are in valid range [0, 1]")

    all_checks = check1 and check2 and check3 and check4

    print("\n" + "="*70)
    if all_checks:
        print("[SUCCESS] ALL VALIDATION CHECKS PASSED!")
        print("Nassar et al. 2021 findings successfully reproduced.")
    else:
        print("[WARNING] SOME VALIDATION CHECKS FAILED")
        print("Review results and compare with original paper.")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Bayesian model fitting against Nassar et al. 2021"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/nassar2021',
        help='Directory containing Nassar .mat files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: output/validation/nassar2021)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Do not generate plots'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR / 'validation' / 'nassar2021'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("NASSAR ET AL. 2021 VALIDATION")
    print("="*70 + "\n")

    # Load data
    loader = NassarDataLoader(args.data_dir)
    data_dict = loader.load_data()

    # Analyze data
    analyzer = NassarAnalysis(data_dict, n_patients=102)

    print("\nExtracting learning rates...")
    lr_subjects = analyzer.extract_learning_rates('subjects')
    lr_model = analyzer.extract_learning_rates('model')

    print("Computing area between curves...")
    area_subjects = analyzer.compute_area_between_curves(lr_subjects)
    area_model = analyzer.compute_area_between_curves(lr_model)

    print("Performing statistical comparisons...")
    stats = analyzer.statistical_comparison(lr_subjects)

    # Print summary
    print_validation_summary(area_subjects, area_model, stats)

    # Save results
    results = {
        'area_subjects': area_subjects,
        'area_model': area_model,
        'statistics': stats
    }

    results_path = output_dir / 'validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved validation results: {results_path}")

    # Generate plots
    if not args.no_plot:
        print("\nGenerating plots...")
        figures_dir = BEHAVIORAL_FIGURES_DIR / 'validation'
        figures_dir.mkdir(parents=True, exist_ok=True)

        plot_figure_6a(lr_subjects, save_path=figures_dir / 'nassar_fig6a_reproduction.png')
        plot_figure_6b(lr_model, save_path=figures_dir / 'nassar_fig6b_reproduction.png')
        plot_area_comparison(area_subjects, area_model,
                           save_path=figures_dir / 'nassar_area_comparison.png')

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
