# Bayesian Normative Models for Predictive Inference

This module provides tools for fitting Bayesian normative models to behavioral data from predictive inference tasks. The models implement optimal Bayesian learning for changepoint and oddball detection scenarios.

## Table of Contents

- [Overview](#overview)
- [Theoretical Background](#theoretical-background)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Parameters](#model-parameters)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [References](#references)

## Overview

The Bayesian normative model computes optimal learning rates based on:
1. **Changepoint probability (Ω)**: Likelihood that the environment has changed
2. **Relative uncertainty (τ)**: Uncertainty about the current state
3. **Learning rate (α)**: Combines Ω and τ depending on task context

Two implementations are available:
- **PyEM** (recommended): Fast optimization using scipy
- **PyMC** (experimental): Full Bayesian inference with MCMC sampling

## Theoretical Background

### Task Description

The **helicopter-bag task** requires predicting where a bag will fall:
- A helicopter at position `h_t` drops bags
- Bags fall at position `b_t ~ N(h_t, σ=20)` (Gaussian noise)
- Agent positions bucket at `a_t` to catch bags

Two experimental conditions:
1. **Changepoint (CP)**: Helicopter position changes abruptly at random trials (hazard rate H)
2. **Oddball (OB)**: Helicopter drifts slowly, but occasional bags fall from random positions (hazard rate H)

### Model Equations

The normative model is defined by seven key equations from Loosen et al. (2023):

#### Equation 1: Normative Update
```
update_t = α_t × δ_t
```
Where:
- `δ_t = b_t - a_t` is the prediction error
- `α_t` is the learning rate

#### Equation 2: Learning Rate (Changepoint)
```
α_t^CP = Ω_t + τ_t - (Ω_t × τ_t)
```

#### Equation 3: Learning Rate (Oddball)
```
α_t^OB = τ_t - (Ω_t × τ_t)
```

#### Equation 4: Changepoint Probability
```
Ω_t = (H × U(δ_t)^LW) / (H × U(δ_t)^LW + (1-H) × N(δ_t | 0, σ_t)^LW)
```
Where:
- `H` = prior hazard rate (frequency of extreme events)
- `U(δ_t)` = uniform PDF (extreme outcomes)
- `N(δ_t | 0, σ_t)` = normal PDF (expected outcomes given current uncertainty)
- `LW` = likelihood weight (extremeness sensitivity)

#### Equation 5: Relative Uncertainty
```
τ_{t+1} = [precision-weighted integration formula] / UU
```
Where:
- `UU` = uncertainty underestimation parameter

#### Equation 6: Likelihood
```
L_t = N(agent_update_t | normative_update_t, σ_update_t)
```

#### Equation 7: Update Variance
```
σ_update_t = σ_motor + |normative_update_t| × σ_LR
```

### Model Parameters

The model has **5 free parameters**:

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Hazard rate | H | [0, 1] | Prior frequency of extreme events |
| Likelihood weight | LW | [0, 1] | How extremeness factors into changepoint detection |
| Uncertainty underestimation | UU | [0, 1] | Inappropriate reduction of uncertainty |
| Motor noise | σ_motor | [0, 5] | Base width of update distribution |
| LR variance slope | σ_LR | [0, 5] | Variance scaling with update size |

## Installation

The Bayesian module is included in the nn4psych package:

```bash
cd /path/to/nn4psych
pip install -e .
```

Dependencies:
- numpy
- scipy
- matplotlib
- pandas
- tqdm

## Quick Start

### Fitting a Single Dataset

```python
import numpy as np
from bayesian import fit_bayesian_model, norm2alpha, norm2beta
from scipy.optimize import minimize

# Load your data
states = np.load('your_data.npy')
bucket_positions = states[1]  # Agent actions
bag_positions = states[2]      # Outcomes

# Set up optimization
initial_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Normalized space

result = minimize(
    fit_bayesian_model,
    initial_params,
    args=(bucket_positions, bag_positions, 'changepoint', None, 'nll'),
    method='Nelder-Mead',
    options={'maxiter': 10000}
)

# Get fitted parameters
fitted_params = result.x
H = norm2alpha(fitted_params[0])
LW = norm2alpha(fitted_params[1])
UU = norm2alpha(fitted_params[2])
sigma_motor = norm2beta(fitted_params[3])
sigma_LR = norm2beta(fitted_params[4])

print(f"Fitted parameters:")
print(f"  H (hazard rate): {H:.3f}")
print(f"  LW (likelihood weight): {LW:.3f}")
print(f"  UU (uncertainty underest.): {UU:.3f}")
print(f"  Negative log-likelihood: {result.fun:.2f}")
```

### Getting Full Model Outputs

```python
# Get trial-by-trial model outputs
model_outputs = fit_bayesian_model(
    fitted_params,
    bucket_positions,
    bag_positions,
    context='changepoint',
    prior=None,
    output='all'
)

# Available outputs:
print(model_outputs.keys())
# ['params', 'bucket_positions', 'bag_positions', 'context',
#  'pred_bucket_placement', 'learning_rate', 'pred_error', 'omega',
#  'tau', 'U_val', 'N_val', 'bucket_update', 'normative_update',
#  'L_normative_update', 'negll', 'BIC']

# Plot learning rate over trials
import matplotlib.pyplot as plt
plt.plot(model_outputs['learning_rate'])
plt.xlabel('Trial')
plt.ylabel('Learning Rate')
plt.title('Model Learning Rate')
plt.show()
```

### Using the Wrapper Script

```bash
# Fit changepoint model
python scripts/fitting/fit_bayesian_pyem.py data/your_data.npy --context changepoint

# Fit oddball model
python scripts/fitting/fit_bayesian_pyem.py data/your_data.npy --context oddball
```

### Batch Processing

```bash
# Fit multiple datasets
python scripts/fitting/batch_fit_bayesian.py data/*.npy --context both --output results/

# Use parallel processing
python scripts/fitting/batch_fit_bayesian.py data/*.npy --parallel --n-jobs 8
```

## Usage Examples

### Example 1: Model Comparison

```python
from bayesian.model_comparison import compare_contexts, print_comparison_summary

# Fit both models
result_cp = minimize(...)  # Changepoint fit
result_ob = minimize(...)  # Oddball fit

# Compare models
comparison = compare_contexts(
    bucket_positions,
    bag_positions,
    result_cp.x,
    result_ob.x
)

# Print results
print_comparison_summary(comparison)
# Output:
# ============================================================
# MODEL COMPARISON SUMMARY
# ============================================================
#
# Negative Log-Likelihood:
#   Changepoint:   245.32
#   Oddball:       289.17
#
# Bayesian Information Criterion (BIC):
#   Changepoint:   513.68
#   Oddball:       601.39
#   Δ BIC (OB-CP):  87.71
#
# Best Model: CHANGEPOINT
# Evidence Ratio: 0.0000
# Evidence Strength: Very strong evidence
# ============================================================
```

### Example 2: Visualization

```python
from bayesian.visualization import (
    plot_model_fit_comprehensive,
    plot_learning_rate_by_prediction_error,
    plot_residuals
)

# Comprehensive 6-panel plot
fig = plot_model_fit_comprehensive(
    model_outputs,
    bucket_positions,
    bag_positions,
    helicopter_positions=states[3],  # Optional
    save_path='figures/model_fit.png'
)

# Learning rate analysis (McGuire et al. 2014 style)
fig = plot_learning_rate_by_prediction_error(
    model_outputs,
    save_path='figures/lr_by_pe.png'
)

# Residual diagnostics
fig = plot_residuals(
    model_outputs,
    save_path='figures/residuals.png'
)
```

### Example 3: Cross-Validation

```python
from bayesian.model_comparison import cross_validate_k_fold

# Perform 5-fold cross-validation
cv_results = cross_validate_k_fold(
    bucket_positions,
    bag_positions,
    context='changepoint',
    initial_params=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    k=5
)

print(f"Cross-validation NegLL: {cv_results['cv_negll_mean']:.2f} ± {cv_results['cv_negll_std']:.2f}")
print(f"Fold-wise NegLL: {cv_results['fold_negll']}")
```

### Example 4: Parameter Recovery

```python
from bayesian.visualization import plot_parameter_distributions

# Simulate data with known parameters and try to recover them
true_params = {'H': 0.125, 'LW': 0.8, 'UU': 0.4, 'sigma_motor': 1.5, 'sigma_LR': 0.5}

# ... simulate data and fit multiple times ...

# Plot recovery
fig = plot_parameter_distributions(
    param_estimates,
    true_values=true_params,
    save_path='figures/param_recovery.png'
)
```

## API Reference

### Core Functions

#### `fit(params, bucket_positions, bag_positions, context, prior=None, output='npl')`

Main fitting function for the Bayesian normative model.

**Parameters:**
- `params` (array): Parameter values in normalized space [H, LW, UU, σ_motor, σ_LR]
- `bucket_positions` (array): Agent's bucket positions over trials
- `bag_positions` (array): Bag landing positions over trials
- `context` (str): 'changepoint' or 'oddball'
- `prior` (callable, optional): Prior function for regularization
- `output` (str): 'npl' (neg posterior likelihood), 'nll' (neg log-likelihood), or 'all' (full outputs)

**Returns:**
- float or dict: Objective value or full model outputs

#### `norm2alpha(x)` and `norm2beta(x, max_val=5)`

Transform normalized parameters to constrained space.

**Parameters:**
- `x` (float or array): Value(s) in normalized space

**Returns:**
- float or array: Transformed values (alpha: [0,1], beta: [0, max_val])

### Model Comparison

#### `calculate_bic(negll, n_params, n_trials)`
Calculate Bayesian Information Criterion.

#### `calculate_aic(negll, n_params)`
Calculate Akaike Information Criterion.

#### `compare_contexts(bucket_positions, bag_positions, fitted_params_cp, fitted_params_ob, prior=None)`
Compare changepoint vs oddball fits.

#### `cross_validate_k_fold(bucket_positions, bag_positions, context, initial_params, k=5, prior=None)`
Perform k-fold cross-validation.

### Visualization

#### `plot_model_fit_comprehensive(results, bucket_positions, bag_positions, helicopter_positions=None, figsize=(16,10), save_path=None)`
Create 6-panel comprehensive visualization.

#### `plot_learning_rate_by_prediction_error(results, bins=10, figsize=(10,6), save_path=None)`
Plot learning rate as function of prediction error (McGuire et al. 2014 style).

#### `plot_parameter_distributions(param_estimates, true_values=None, figsize=(14,8), save_path=None)`
Plot parameter distributions from multiple fits.

#### `plot_model_comparison(comparison, figsize=(12,5), save_path=None)`
Visualize model comparison results.

#### `plot_residuals(results, figsize=(12,4), save_path=None)`
Plot diagnostic residual plots.

## Data Format

Expected data format for `.npy` files:

```python
# Minimum format (shape: [3, n_trials])
states = np.array([
    trials,              # Trial numbers [0, 1, 2, ...]
    bucket_positions,    # Agent's bucket positions
    bag_positions,       # Bag landing positions
])

# Extended format (shape: [5, n_trials])
states = np.array([
    trials,              # Trial numbers
    bucket_positions,    # Agent positions
    bag_positions,       # Bag positions
    helicopter_positions,# True helicopter positions
    hazard_triggers,     # Boolean: whether hazard occurred
])
```

## Output Files

### Individual Fitting (`fit_bayesian_pyem.py`)

- `output/bayesian_fits/pyem_params_{context}.npy`: Fitted parameters
- `output/bayesian_fits/pyem_full_results_{context}.npy`: Full results dict
- `figures/behavioral/pyem_fit_{context}.png`: Visualization

### Batch Fitting (`batch_fit_bayesian.py`)

- `output/bayesian_fits/batch/summary_{context}.csv`: Summary table
- `output/bayesian_fits/batch/full_results_{context}.npy`: All results
- `output/bayesian_fits/batch/param_distributions_{context}.png`: Parameter distributions
- `output/bayesian_fits/batch/plots/{dataset}_{context}.png`: Individual fit plots

## References

1. **Loosen, A. M., Skvortsova, V., & Hauser, T. U. (2023)**. *pyEM: A Python package for EM-based Bayesian estimation of hierarchical cognitive models.* Behavior Research Methods. [https://link.springer.com/article/10.3758/s13428-024-02427-y](https://link.springer.com/article/10.3758/s13428-024-02427-y)

2. **McGuire, J. T., Nassar, M. R., Gold, J. I., & Kable, J. W. (2014)**. *Functionally dissociable influences on learning rate in a dynamic environment.* Neuron, 84(4), 870-881.

3. **Nassar, M. R., Wilson, R. C., Heasly, B., & Gold, J. I. (2010)**. *An approximately Bayesian delta-rule model explains the dynamics of belief updating in a changing environment.* Journal of Neuroscience, 30(37), 12366-12378.

4. **Burnham, K. P., & Anderson, D. R. (2002)**. *Model selection and multimodel inference: A practical information-theoretic approach* (2nd ed.). Springer.

5. **Kass, R. E., & Raftery, A. E. (1995)**. *Bayes factors.* Journal of the American Statistical Association, 90(430), 773-795.

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'bayesian'`
- **Solution**: Make sure you've installed the package: `pip install -e .` from the project root

**Issue**: Parameters all converge to 0.5 or 2.5 (middle values)
- **Solution**: You're getting the un-optimized values. Make sure you're using `minimize()` to optimize, not just calling `fit()` with output='all'

**Issue**: Optimization fails or returns huge negative log-likelihood
- **Solution**: Try different initial parameter values or check that your data is in the correct format

**Issue**: Plots not saving
- **Solution**: Check that the output directory exists and you have write permissions

## Contributing

To add new features or fix bugs:
1. Follow the existing code style
2. Add docstrings to all functions
3. Include equation references where applicable
4. Test with sample data before committing

## License

See project LICENSE file.
