#!/usr/bin/env python3
"""
Extract Trial-by-Trial Data from Nassar et al. 2021

This script extracts and cleans trial-by-trial behavioral data from the Nassar
et al. (2021) dataset, applying the same data preprocessing procedures used in
their analysis to ensure fair comparison.

Data Cleaning Procedures (from AASP_mastList.m):
1. Drop first 3 trials from each 100-trial block
2. Exclude trials with non-finite updates or prediction errors
3. Cap learning rates at [0, 1]
4. Concatenate across 4 conditions per subject

Usage:
    python extract_nassar_trials.py

Output:
    data/processed/nassar2021/
        - subject_trials.npy: All subjects' trial data
        - subject_metadata.csv: Subject info (patient/control, ID)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from typing import Dict, List, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import OUTPUT_DIR

# Nassar data directory
NASSAR_DIR = Path('C:/Users/aman0087/Documents/Github/Nassar_et_al_2021/Brain2021Code')
REAL_SUBJECTS_DIR = NASSAR_DIR / 'realSubjects'

# Data cleaning parameters (from AASP_mastList.m)
DROP_FIRST_N_TRIALS = 3  # drop=3
EXPECTED_TRIALS_PER_BLOCK = 100
EXPECTED_BLOCKS = 100  # Per condition

# Task conditions
CONDITIONS = [
    'cloud_cp_avoid',
    'cloud_cp_seek',
    'cloud_drift_avoid',
    'cloud_drift_seek'
]


def load_subject_file(file_path: Path) -> Dict:
    """
    Load a single subject .mat file.

    Returns
    -------
    dict with 'statusData' and 'payoutData'
    """
    data = sio.loadmat(str(file_path))
    return data


def extract_trial_data_from_blocks(status_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract trial-by-trial data from statusData blocks.

    Parameters
    ----------
    status_data : np.ndarray
        Array of 100 blocks (each block is a trial)

    Returns
    -------
    dict
        Trial-by-trial data arrays
    """
    n_trials = len(status_data)

    # Initialize arrays
    outcome = np.zeros(n_trials)           # bag position (currentOutcome)
    prediction = np.zeros(n_trials)        # bucket position (currentPrediction)
    update = np.zeros(n_trials)            # bucket update (currentUpdate)
    delta = np.zeros(n_trials)             # prediction error (currentDelta)
    trial_num = np.zeros(n_trials)         # trial number in block
    heli_pos = np.zeros(n_trials)          # helicopter position (currentMean)
    is_change_trial = np.zeros(n_trials, dtype=bool)  # changepoint trial

    # Extract data from each block
    for i in range(n_trials):
        block = status_data[i]

        # Extract scalar values - handle MATLAB nested structure
        # Values are stored as scalar arrays, need to extract the actual value
        try:
            outcome[i] = np.asarray(block['currentOutcome']).flatten()[0]
            prediction[i] = np.asarray(block['currentPrediction']).flatten()[0]
            update[i] = np.asarray(block['currentUpdate']).flatten()[0]
            delta[i] = np.asarray(block['currentDelta']).flatten()[0]
            trial_num[i] = np.asarray(block['blockCompletedTrials']).flatten()[0]
            heli_pos[i] = np.asarray(block['currentMean']).flatten()[0]

            # Boolean flag
            is_change_trial[i] = bool(np.asarray(block['isChangeTrial']).flatten()[0])
        except Exception as e:
            # If extraction fails, use NaN values (will be filtered later)
            outcome[i] = np.nan
            prediction[i] = np.nan
            update[i] = np.nan
            delta[i] = np.nan
            trial_num[i] = np.nan
            heli_pos[i] = np.nan
            is_change_trial[i] = False

    return {
        'outcome': outcome,
        'prediction': prediction,
        'update': update,
        'delta': delta,
        'trial_num': trial_num,
        'heli_pos': heli_pos,
        'is_change_trial': is_change_trial,
    }


def load_subject_data(subject_dir: Path) -> Dict[str, Dict]:
    """
    Load all condition files for a single subject.

    Parameters
    ----------
    subject_dir : Path
        Path to subject directory

    Returns
    -------
    dict
        Data for each condition
    """
    subject_id = subject_dir.name
    subject_data = {}

    print(f"\nLoading subject: {subject_id}")

    for condition in CONDITIONS:
        # Find file for this condition
        files = list(subject_dir.glob(f"{subject_id}_{condition}_*.mat"))

        # Filter out timing files
        files = [f for f in files if 'topsDataLog' not in f.name]

        if len(files) == 0:
            print(f"  WARNING: No file found for {condition}")
            continue

        if len(files) > 1:
            print(f"  WARNING: Multiple files for {condition}, using first")

        # Load the data
        file_path = files[0]
        print(f"  {condition}: {file_path.name}")

        try:
            data = load_subject_file(file_path)
            status_data = data['statusData']

            # Verify expected structure
            if len(status_data) != EXPECTED_BLOCKS:
                print(f"    WARNING: Expected {EXPECTED_BLOCKS} blocks, got {len(status_data)}")

            # Extract trial-by-trial data
            trial_data = extract_trial_data_from_blocks(status_data)
            subject_data[condition] = trial_data

        except Exception as e:
            print(f"    ERROR loading {condition}: {e}")
            continue

    return subject_data


def apply_nassar_cleaning(trial_data: Dict[str, np.ndarray],
                          drop_n: int = DROP_FIRST_N_TRIALS) -> Tuple[np.ndarray, Dict]:
    """
    Apply Nassar et al. data cleaning procedures.

    From AASP_mastList.m line 476:
        isGood = all(isfinite(xes2), 2) & isfinite(PE) & isfinite(UP) & trialNum'>drop

    Parameters
    ----------
    trial_data : dict
        Raw trial data
    drop_n : int
        Number of initial trials to drop

    Returns
    -------
    valid_mask : np.ndarray
        Boolean mask for valid trials
    stats : dict
        Statistics about excluded trials
    """
    update = trial_data['update']
    delta = trial_data['delta']
    trial_num = trial_data['trial_num']

    # Apply Nassar criteria
    is_finite_update = np.isfinite(update)
    is_finite_delta = np.isfinite(delta)
    is_after_drop = trial_num > drop_n

    # Combined validity mask
    valid_mask = is_finite_update & is_finite_delta & is_after_drop

    # Compute statistics
    stats = {
        'total_trials': len(update),
        'non_finite_update': (~is_finite_update).sum(),
        'non_finite_delta': (~is_finite_delta).sum(),
        'dropped_early': (~is_after_drop).sum(),
        'valid_trials': valid_mask.sum(),
        'excluded_trials': (~valid_mask).sum(),
    }

    return valid_mask, stats


def concatenate_conditions(subject_data: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """
    Concatenate data across all 4 conditions for a subject.

    Order matches AASP_mastList.m lines 221-239:
        [cloud_cp_avoid, cloud_cp_seek, cloud_drift_avoid, cloud_drift_seek]
    """
    # Initialize lists
    all_outcome = []
    all_prediction = []
    all_update = []
    all_delta = []
    all_trial_num = []
    all_condition = []

    condition_map = {
        'cloud_cp_avoid': 0,
        'cloud_cp_seek': 1,
        'cloud_drift_avoid': 2,
        'cloud_drift_seek': 3,
    }

    # Concatenate in order
    for condition in CONDITIONS:
        if condition not in subject_data:
            continue

        data = subject_data[condition]
        n_trials = len(data['outcome'])

        all_outcome.append(data['outcome'])
        all_prediction.append(data['prediction'])
        all_update.append(data['update'])
        all_delta.append(data['delta'])
        all_trial_num.append(data['trial_num'])
        all_condition.append(np.full(n_trials, condition_map[condition]))

    return {
        'outcome': np.concatenate(all_outcome),
        'prediction': np.concatenate(all_prediction),
        'update': np.concatenate(all_update),
        'delta': np.concatenate(all_delta),
        'trial_num': np.concatenate(all_trial_num),
        'condition': np.concatenate(all_condition),
    }


def extract_all_subjects():
    """
    Extract trial data for all subjects (patients and controls).
    """
    print("=" * 70)
    print("EXTRACTING NASSAR ET AL. 2021 TRIAL-BY-TRIAL DATA")
    print("=" * 70)

    # Find all subject directories
    patient_dir = REAL_SUBJECTS_DIR / 'Patients'
    patient2_dir = REAL_SUBJECTS_DIR / 'Patients2'
    control_dir = REAL_SUBJECTS_DIR / 'Normal Controls'

    patient_dirs = sorted(list(patient_dir.glob('SP_*')))
    patient2_dirs = sorted(list(patient2_dir.glob('SP_*')))
    control_dirs = sorted(list(control_dir.glob('SP_*')))

    all_dirs = patient_dirs + patient2_dirs + control_dirs

    print(f"\nFound {len(patient_dirs)} patients (cohort 1)")
    print(f"Found {len(patient2_dirs)} patients (cohort 2)")
    print(f"Found {len(control_dirs)} controls")
    print(f"Total: {len(all_dirs)} subjects")

    # Process all subjects
    all_subjects_data = []
    metadata_rows = []

    for i, subject_dir in enumerate(all_dirs):
        subject_id = subject_dir.name
        is_patient = (subject_dir.parent.name in ['Patients', 'Patients2'])

        # Load subject data
        subject_data = load_subject_data(subject_dir)

        if len(subject_data) == 0:
            print(f"  WARNING: No valid conditions for {subject_id}, skipping")
            continue

        # Concatenate across conditions
        concat_data = concatenate_conditions(subject_data)

        # Apply Nassar cleaning
        valid_mask, stats = apply_nassar_cleaning(concat_data)

        print(f"\n  Data cleaning statistics:")
        print(f"    Total trials: {stats['total_trials']}")
        print(f"    Dropped (first {DROP_FIRST_N_TRIALS} per block): {stats['dropped_early']}")
        print(f"    Non-finite updates: {stats['non_finite_update']}")
        print(f"    Non-finite deltas: {stats['non_finite_delta']}")
        print(f"    Valid trials: {stats['valid_trials']}")

        # Store data
        subject_trial_data = {
            'subject_id': subject_id,
            'is_patient': is_patient,
            'outcome': concat_data['outcome'][valid_mask],  # bag positions
            'prediction': concat_data['prediction'][valid_mask],  # bucket positions
            'update': concat_data['update'][valid_mask],
            'delta': concat_data['delta'][valid_mask],
            'condition': concat_data['condition'][valid_mask],
            'n_trials': valid_mask.sum(),
        }

        all_subjects_data.append(subject_trial_data)

        # Metadata
        metadata_rows.append({
            'subject_id': subject_id,
            'subject_num': i,
            'is_patient': is_patient,
            'n_trials': valid_mask.sum(),
            'n_excluded': stats['excluded_trials'],
        })

    print(f"\n{'='*70}")
    print(f"Successfully extracted data for {len(all_subjects_data)} subjects")
    print(f"{'='*70}")

    # Save data
    output_dir = OUTPUT_DIR / 'processed' / 'nassar2021'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trial data
    trial_data_path = output_dir / 'subject_trials.npy'
    np.save(trial_data_path, all_subjects_data, allow_pickle=True)
    print(f"\nSaved trial data to: {trial_data_path}")

    # Save metadata
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = output_dir / 'subject_metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Saved metadata to: {metadata_path}")

    # Print summary statistics
    print(f"\n[SUMMARY]")
    print(f"Patients: {metadata_df['is_patient'].sum()}")
    print(f"Controls: {(~metadata_df['is_patient']).sum()}")
    print(f"Mean trials per subject: {metadata_df['n_trials'].mean():.1f}")
    print(f"Median trials per subject: {metadata_df['n_trials'].median():.1f}")

    return all_subjects_data, metadata_df


if __name__ == "__main__":
    extract_all_subjects()
