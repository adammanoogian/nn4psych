#!/usr/bin/env python3
"""Inspect Nassar data structure to understand what we have."""

import scipy.io as sio
import numpy as np
from pathlib import Path

# Load both files
model_path = 'data/raw/nassar2021/slidingWindowFits_model_23-Nov-2021.mat'
subject_path = 'data/raw/nassar2021/slidingWindowFits_subjects_23-Nov-2021.mat'

print("=" * 60)
print("NASSAR DATA INSPECTION")
print("=" * 60)

# Model data
print("\n[MODEL DATA]")
model_data = sio.loadmat(model_path)
print(f"File: {model_path}")
print(f"Keys: {list(model_data.keys())}")
print(f"\nbinRegData type: {type(model_data['binRegData'])}")
print(f"binRegData shape: {np.asarray(model_data['binRegData']).shape}")

# Subject data
print("\n[SUBJECT DATA]")
subject_data = sio.loadmat(subject_path)
print(f"File: {subject_path}")
print(f"Keys: {list(subject_data.keys())}")
print(f"\nbinRegData type: {type(subject_data['binRegData'])}")
print(f"binRegData shape: {np.asarray(subject_data['binRegData']).shape}")

# Detailed structure
print("\n[DATA STRUCTURE]")
model_arr = np.asarray(model_data['binRegData'])
subject_arr = np.asarray(subject_data['binRegData'])

print(f"\nmodel_arr.shape: {model_arr.shape}")
print(f"subject_arr.shape: {subject_arr.shape}")

# Access pattern: data[0][0][condition][participant]
print("\n[CONDITION/PARTICIPANT STRUCTURE]")
print(f"Number of conditions: {len(model_arr[0][0])}")
print(f"Number of participants (model): {len(model_arr[0][0][0])}")
print(f"Number of participants (subject): {len(subject_arr[0][0][0])}")

# Check a single participant's data
print("\n[SINGLE PARTICIPANT DATA]")
# CP condition, first patient
cp_data = subject_arr[0][0][0][0]
print(f"CP condition, Patient 0 shape: {cp_data.shape}")
print(f"CP condition, Patient 0 columns: {cp_data.shape[1]}")
print(f"Sample data (first 5 bins):")
print(f"  Bin centers (col 0): {cp_data[:5, 0]}")
print(f"  Learning rates (col 1): {cp_data[:5, 1]}")

print("\n[SUMMARY]")
print("Data format: Sliding window fits (binned by prediction error)")
print("  - 115 bins per participant per condition")
print("  - Column 0: Bin centers (prediction error magnitude)")
print("  - Column 1: Learning rate values")
print("\nPROBLEM: This is aggregated data, not trial-by-trial!")
print("  Need: bucket_positions, bag_positions for each trial")
print("  Have: Learning rates binned by prediction error magnitude")

print("\n" + "=" * 60)
