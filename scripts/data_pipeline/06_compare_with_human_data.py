#!/usr/bin/env python3
"""
Stage 06: Compare with Human Data (Nassar 2021)

Replicates Nassar 2021 Figure 6 comparing model and human learning behaviors.
Compares patients vs controls in change-point and oddball conditions.

Input: data/raw/nassar2021/*.mat files
Output: Figures in figures/behavioral_summary/ and figures/model_performance/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from config import (
    RAW_DATA_DIR, BEHAVIORAL_FIGURES_DIR, MODEL_PERFORMANCE_FIGURES_DIR
)


def load_nassar_data(data_dir):
    """Load Nassar 2021 sliding window fits data."""
    data_dir = Path(data_dir)

    mod_file = data_dir / 'nassar2021' / 'slidingWindowFits_model_23-Nov-2021.mat'
    sub_file = data_dir / 'nassar2021' / 'slidingWindowFits_subjects_23-Nov-2021.mat'

    if not mod_file.exists() or not sub_file.exists():
        raise FileNotFoundError(f"Nassar data files not found in {data_dir / 'nassar2021'}")

    mod_data = np.asarray(sio.loadmat(str(mod_file))['binRegData'])
    sub_data = np.asarray(sio.loadmat(str(sub_file))['binRegData'])

    return mod_data, sub_data


def compute_statistics(data, sz_pat=102):
    """Compute mean and std for patients and controls."""
    # Data structure: data[0][0][condition][participant][:][1]
    # condition: 0=CP, 1=OB
    cp, ob = 0, 1

    stats = {
        'sub': {
            'cp_pat': {'mean': np.mean(data[0][0][cp][:sz_pat], axis=0),
                      'std': np.std(data[0][0][cp][:sz_pat], axis=0)},
            'ob_pat': {'mean': np.mean(data[0][0][ob][:sz_pat], axis=0),
                      'std': np.std(data[0][0][ob][:sz_pat], axis=0)},
            'cp_con': {'mean': np.mean(data[0][0][cp][sz_pat:], axis=0),
                      'std': np.std(data[0][0][cp][sz_pat:], axis=0)},
            'ob_con': {'mean': np.mean(data[0][0][ob][sz_pat:], axis=0),
                      'std': np.std(data[0][0][ob][sz_pat:], axis=0)},
        }
    }

    # Calculate number of controls
    n_controls = len(data[0][0][cp]) - sz_pat
    stats['n_patients'] = sz_pat
    stats['n_controls'] = n_controls

    return stats


def compute_area_between_curves(cp_curve, ob_curve):
    """Compute area between CP and OB learning rate curves."""
    return np.trapz(np.abs(cp_curve - ob_curve), dx=1)


def plot_nassar_fig6a(sub_stats, output_dir, show_diff=True):
    """Plot Figure 6a: Subject learning rates by condition."""
    output_dir = Path(output_dir)
    x_axis = np.arange(115)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim(0, 1)

    # Plot subject data (column 1 is learning rate)
    ax.plot(x_axis, sub_stats['cp_pat']['mean'][:, 1], label='CP Patients', color='orange')
    ax.plot(x_axis, sub_stats['ob_pat']['mean'][:, 1], label='OB Patients', color='brown')
    ax.plot(x_axis, sub_stats['cp_con']['mean'][:, 1], label='CP Controls', color='blue')
    ax.plot(x_axis, sub_stats['ob_con']['mean'][:, 1], label='OB Controls', color='green')

    if show_diff:
        diff_pat = np.mean([sub_stats['cp_pat']['mean'][:, 1],
                           sub_stats['ob_pat']['mean'][:, 1]], axis=0)
        diff_con = np.mean([sub_stats['cp_con']['mean'][:, 1],
                           sub_stats['ob_con']['mean'][:, 1]], axis=0)
        ax.plot(x_axis, diff_pat, label='Mean Patients', color='red', linestyle='--')
        ax.plot(x_axis, diff_con, label='Mean Controls', color='purple', linestyle='--')

    ax.set_xlabel('Relative Error')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Subject Learning Rates (Nassar 2021 Fig 6a)')
    ax.legend()
    plt.tight_layout()

    plt.savefig(output_dir / 'nassarfig6a.png', dpi=150)
    plt.savefig(output_dir / 'nassarfig6a.svg')
    plt.close()


def plot_nassar_fig6b(mod_stats, output_dir, show_diff=True):
    """Plot Figure 6b: Model learning rates by condition."""
    output_dir = Path(output_dir)
    x_axis = np.arange(115)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim(0, 1)

    ax.plot(x_axis, mod_stats['cp_pat']['mean'][:, 1], label='CP Patients', color='orange')
    ax.plot(x_axis, mod_stats['ob_pat']['mean'][:, 1], label='OB Patients', color='brown')
    ax.plot(x_axis, mod_stats['cp_con']['mean'][:, 1], label='CP Controls', color='blue')
    ax.plot(x_axis, mod_stats['ob_con']['mean'][:, 1], label='OB Controls', color='green')

    if show_diff:
        diff_pat = np.mean([mod_stats['cp_pat']['mean'][:, 1],
                           mod_stats['ob_pat']['mean'][:, 1]], axis=0)
        diff_con = np.mean([mod_stats['cp_con']['mean'][:, 1],
                           mod_stats['ob_con']['mean'][:, 1]], axis=0)
        ax.plot(x_axis, diff_pat, label='Mean Patients', color='red', linestyle='--')
        ax.plot(x_axis, diff_con, label='Mean Controls', color='purple', linestyle='--')

    ax.set_xlabel('Relative Error')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Model Learning Rates (Nassar 2021 Fig 6b)')
    ax.legend()
    plt.tight_layout()

    plt.savefig(output_dir / 'nassarfig6b.png', dpi=150)
    plt.savefig(output_dir / 'nassarfig6b.svg')
    plt.close()


def plot_area_comparison(sub_stats, mod_stats, output_dir):
    """Plot bar graph comparing area between curves."""
    output_dir = Path(output_dir)

    # Calculate areas
    area_sub_pat = compute_area_between_curves(
        sub_stats['cp_pat']['mean'][:, 1], sub_stats['ob_pat']['mean'][:, 1])
    area_sub_con = compute_area_between_curves(
        sub_stats['cp_con']['mean'][:, 1], sub_stats['ob_con']['mean'][:, 1])
    area_mod_pat = compute_area_between_curves(
        mod_stats['cp_pat']['mean'][:, 1], mod_stats['ob_pat']['mean'][:, 1])
    area_mod_con = compute_area_between_curves(
        mod_stats['cp_con']['mean'][:, 1], mod_stats['ob_con']['mean'][:, 1])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ['Subject\nPatients', 'Subject\nControls', 'Model\nPatients', 'Model\nControls'],
        [area_sub_pat, area_sub_con, area_mod_pat, area_mod_con],
        color=['orange', 'blue', 'coral', 'steelblue']
    )

    ax.set_ylabel('Area Between Curves')
    ax.set_title('CP vs OB Learning Rate Difference')
    plt.tight_layout()

    plt.savefig(output_dir / 'area_between_curves.png', dpi=150)
    plt.savefig(output_dir / 'area_between_curves.svg')
    plt.close()

    return {
        'sub_pat': area_sub_pat, 'sub_con': area_sub_con,
        'mod_pat': area_mod_pat, 'mod_con': area_mod_con
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 06: Compare with Human Data")
    parser.add_argument('--data_dir', type=str, default=str(RAW_DATA_DIR),
                       help='Directory containing Nassar data')
    parser.add_argument('--output_dir', type=str, default=str(BEHAVIORAL_FIGURES_DIR),
                       help='Output directory for figures')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    perf_dir = MODEL_PERFORMANCE_FIGURES_DIR
    perf_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 06: COMPARE WITH HUMAN DATA")
    print("=" * 60)

    print("\n1. Loading Nassar 2021 data...")
    try:
        mod_data, sub_data = load_nassar_data(args.data_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure data/raw/nassar2021/ contains the .mat files")
        return 1

    print("\n2. Computing statistics...")
    sub_stats = compute_statistics(sub_data)['sub']
    mod_stats = compute_statistics(mod_data)['sub']

    print("\n3. Generating figures...")
    plot_nassar_fig6a(sub_stats, output_dir)
    print(f"   Saved nassarfig6a to {output_dir}")

    plot_nassar_fig6b(mod_stats, output_dir)
    print(f"   Saved nassarfig6b to {output_dir}")

    areas = plot_area_comparison(sub_stats, mod_stats, perf_dir)
    print(f"   Saved area_between_curves to {perf_dir}")

    print("\n4. Summary:")
    print(f"   Subject Patients Area: {areas['sub_pat']:.2f}")
    print(f"   Subject Controls Area: {areas['sub_con']:.2f}")
    print(f"   Model Patients Area: {areas['mod_pat']:.2f}")
    print(f"   Model Controls Area: {areas['mod_con']:.2f}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
