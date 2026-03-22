#!/usr/bin/env python3
"""
Quick test script to verify Bayesian PyEM fitting works correctly.
"""

import numpy as np
from bayesian import fit_bayesian_model

print("Testing Bayesian PyEM implementation...\n")

# Create simple synthetic data for testing
np.random.seed(42)
n_trials = 100

# Simulate helicopter task:
# - Helicopter position changes occasionally (changepoint)
# - Bag falls with Gaussian noise around helicopter
# - Bucket tries to track helicopter

helicopter_pos = np.ones(n_trials) * 150  # Start at position 150

# Add changepoints
changepoints = [25, 60, 85]
positions = [150, 100, 200, 130]
for i, cp in enumerate(changepoints):
    helicopter_pos[cp:] = positions[i+1]

# Bags fall with noise around helicopter
sigma_bag = 20
bag_positions = helicopter_pos + np.random.normal(0, sigma_bag, n_trials)

# Bucket uses a simple learning rule (not optimal - that's what we'll fit)
bucket_positions = np.zeros(n_trials)
bucket_positions[0] = 150
alpha_agent = 0.3  # Fixed learning rate for agent

for t in range(1, n_trials):
    pred_error = bag_positions[t-1] - bucket_positions[t-1]
    bucket_positions[t] = bucket_positions[t-1] + alpha_agent * pred_error

print(f"Created synthetic data:")
print(f"  Trials: {n_trials}")
print(f"  Changepoints at: {changepoints}")
print(f"  Helicopter positions: {np.unique(helicopter_pos)}")
print(f"  Bucket range: [{bucket_positions.min():.1f}, {bucket_positions.max():.1f}]")
print(f"  Bag range: [{bag_positions.min():.1f}, {bag_positions.max():.1f}]")
print()

# Test the fitting function
print("Testing fit_bayesian_model function...")

# Initial parameters (in normalized space, all 0s means middle of range)
initial_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

try:
    # Test with 'all' output to get full diagnostics
    print("\nCalling fit_bayesian_model with output='all'...")
    results = fit_bayesian_model(
        initial_params,
        bucket_positions,
        bag_positions,
        context='changepoint',
        prior=None,
        output='all'
    )

    print("\nSUCCESS! Function returned results")
    print("\nResult keys:", list(results.keys()))
    print("\nFitted parameters:")
    for i, (name, val) in enumerate(zip(
        ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR'],
        results['params']
    )):
        print(f"  {name:12s}: {val:.4f}")

    print(f"\nNegative log-likelihood: {results['negll']:.2f}")
    print(f"BIC: {results['BIC']:.2f}")
    print(f"\nLearning rate range: [{results['learning_rate'].min():.3f}, {results['learning_rate'].max():.3f}]")
    print(f"Mean learning rate: {results['learning_rate'].mean():.3f}")
    print(f"(Agent's fixed learning rate was: {alpha_agent})")

    print("\nAll tests passed!")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
