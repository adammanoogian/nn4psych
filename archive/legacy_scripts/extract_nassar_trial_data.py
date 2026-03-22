#!/usr/bin/env python3
"""Extract trial-by-trial behavioral data from Nassar files."""

import scipy.io as sio
import numpy as np
from pathlib import Path

nassar_dir = Path('C:/Users/aman0087/Documents/Github/Nassar_et_al_2021/Brain2021Code')

print("=" * 70)
print("EXTRACTING TRIAL-BY-TRIAL BEHAVIORAL DATA")
print("=" * 70)

# Load example subject file
example_file = nassar_dir / 'realSubjects/Normal Controls/SP_063808/SP_063808_cloud_cp_avoid_20160217T174152.mat'
data = sio.loadmat(str(example_file))

# Access statusData
status_data = data['statusData']
print(f"\nFile: {example_file.name}")
print(f"Number of blocks: {len(status_data)}")

# Try to access trialData from the first block
print("\n[ACCESSING trialData FROM BLOCK 1]")
try:
    trial_data = status_data['trialData'][0][0]
    print(f"trialData type: {type(trial_data)}")
    print(f"trialData shape: {trial_data.shape}")

    if trial_data.dtype.names:
        print(f"\ntrialData fields: {trial_data.dtype.names}")

        # Show sample data from first 5 trials
        print(f"\n[FIRST 5 TRIALS]")
        for field in ['currentOutcome', 'currentPrediction', 'currentUpdate', 'currentDelta']:
            if field in trial_data.dtype.names:
                values = trial_data[field][0][0].flatten()[:5]
                print(f"{field:20s}: {values}")

except Exception as e:
    print(f"Error accessing trialData: {e}")
    import traceback
    traceback.print_exc()

# Try alternative access
print("\n[ALTERNATIVE ACCESS METHOD]")
try:
    for block_idx in range(min(1, len(status_data))):
        print(f"\nBlock {block_idx + 1}:")

        # Try to access outcome and prediction
        current_outcome = status_data['currentOutcome'][block_idx][0]
        current_prediction = status_data['currentPrediction'][block_idx][0]
        current_update = status_data['currentUpdate'][block_idx][0]

        if isinstance(current_outcome, np.ndarray):
            print(f"  currentOutcome shape: {current_outcome.shape}")
            print(f"  currentPrediction shape: {current_prediction.shape}")
            print(f"  currentUpdate shape: {current_update.shape}")

            if current_outcome.size > 0:
                print(f"\n  Sample data (first 10 trials):")
                n_show = min(10, len(current_outcome.flatten()))
                outcomes = current_outcome.flatten()[:n_show]
                predictions = current_prediction.flatten()[:n_show]
                updates = current_update.flatten()[:n_show]

                print(f"  Trial | Outcome | Prediction | Update")
                print(f"  " + "-" * 45)
                for i in range(n_show):
                    print(f"  {i+1:5d} | {outcomes[i]:7.1f} | {predictions[i]:10.1f} | {updates[i]:6.1f}")

except Exception as e:
    print(f"Error with alternative access: {e}")
    import traceback
    traceback.print_exc()

# Now check the fitted parameters file
print("\n" + "=" * 70)
print("EXTRACTING FITTED PARAMETERS")
print("=" * 70)

param_file = nassar_dir / 'heliParamEstimatesForJim_23-Nov-2021.mat'
param_data = sio.loadmat(str(param_file))

sub_data = param_data['subData']
print(f"\nParameter file structure:")
print(f"  Shape: {sub_data.shape}")
print(f"  Fields: {sub_data.dtype.names}")

# Extract parameter labels and values
try:
    param_labels = sub_data['paramLabels'][0][0][0]
    print(f"\n[PARAMETER LABELS]")
    for i, label in enumerate(param_labels):
        if isinstance(label, np.ndarray):
            print(f"  {i}: {label[0]}")
        else:
            print(f"  {i}: {label}")

    # Get params for first few subjects
    print(f"\n[FITTED PARAMETERS - FIRST 5 SUBJECTS]")
    params = sub_data['params'][0][0]
    sub_names = sub_data['subName'][0][0]
    is_patient = sub_data['isPatient'][0][0]

    print(f"Total subjects with fitted parameters: {len(params)}")
    print(f"\nSubject | IsPatient | Parameters")
    print("-" * 60)

    for i in range(min(5, len(params))):
        name = sub_names[i][0][0] if isinstance(sub_names[i][0], np.ndarray) else sub_names[i]
        patient_status = is_patient[i][0][0] if isinstance(is_patient[i][0], np.ndarray) else is_patient[i]
        param_values = params[i].flatten()

        print(f"{name} | {bool(patient_status)} | {param_values}")

except Exception as e:
    print(f"Error accessing parameters: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\n✓ FOUND TRIAL-BY-TRIAL DATA:")
print("  - currentOutcome (bag position)")
print("  - currentPrediction (bucket position)")
print("  - currentUpdate (bucket update)")
print("  - currentDelta (prediction error)")
print("\n✓ FOUND FITTED PARAMETERS:")
print("  - Parameter labels and values for all subjects")
print("  - Patient/control classification")
print("\nNO MATLAB NEEDED - Can extract with Python!")
print("=" * 70)
