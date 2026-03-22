#!/usr/bin/env python3
"""Inspect raw Nassar subject data to see if we have trial-by-trial behavioral data."""

import scipy.io as sio
import numpy as np
from pathlib import Path

# Path to Nassar code directory
nassar_dir = Path('C:/Users/aman0087/Documents/Github/Nassar_et_al_2021/Brain2021Code')

# Check one example subject's data
example_file = nassar_dir / 'realSubjects/Normal Controls/SP_063808/SP_063808_cloud_cp_avoid_20160217T174152.mat'

print("=" * 70)
print("INSPECTING RAW NASSAR SUBJECT DATA")
print("=" * 70)
print(f"\nFile: {example_file.name}")

if not example_file.exists():
    print(f"ERROR: File not found: {example_file}")
    exit(1)

# Load the file
try:
    data = sio.loadmat(str(example_file))
    print(f"\n[SUCCESS] File loaded!")

    # Show all keys
    print(f"\n[KEYS IN FILE]")
    keys = [k for k in data.keys() if not k.startswith('__')]
    for key in keys:
        print(f"  - {key}: {type(data[key])}")

    # Try to find relevant fields
    print(f"\n[SEARCHING FOR BEHAVIORAL DATA]")

    for key in keys:
        val = data[key]
        if isinstance(val, np.ndarray):
            print(f"\n{key}:")
            print(f"  Type: {type(val)}")
            print(f"  Shape: {val.shape}")
            print(f"  Dtype: {val.dtype}")

            # Check if it's a structured array
            if val.dtype.names:
                print(f"  Fields: {val.dtype.names}")

                # Check for promising field names
                promising_fields = []
                for field in val.dtype.names:
                    field_lower = field.lower()
                    if any(keyword in field_lower for keyword in
                           ['bucket', 'bag', 'helicopter', 'cloud', 'position',
                            'outcome', 'update', 'trial', 'heli', 'pred']):
                        promising_fields.append(field)

                if promising_fields:
                    print(f"  *** PROMISING FIELDS FOUND: {promising_fields}")

                    # Try to access first few values
                    for field in promising_fields[:5]:  # Limit to first 5
                        try:
                            field_data = val[field][0][0]
                            if isinstance(field_data, np.ndarray):
                                print(f"\n    {field}:")
                                print(f"      Shape: {field_data.shape}")
                                print(f"      First 5 values: {field_data.flatten()[:5]}")
                        except Exception as e:
                            print(f"    {field}: Could not access ({e})")

    # Check if there's a heliParamEstimates file with fitted parameters
    print("\n" + "=" * 70)
    print("CHECKING FOR FITTED PARAMETERS")
    print("=" * 70)

    param_file = nassar_dir / 'heliParamEstimatesForJim_23-Nov-2021.mat'
    if param_file.exists():
        print(f"\nFound parameter file: {param_file.name}")
        param_data = sio.loadmat(str(param_file))

        param_keys = [k for k in param_data.keys() if not k.startswith('__')]
        print(f"Keys: {param_keys}")

        for key in param_keys:
            val = param_data[key]
            if isinstance(val, np.ndarray):
                print(f"\n{key}:")
                print(f"  Shape: {val.shape}")
                if val.dtype.names:
                    print(f"  Fields: {val.dtype.names}")

    print("\n" + "=" * 70)

except Exception as e:
    print(f"\n[ERROR] Failed to load file: {e}")
    import traceback
    traceback.print_exc()
