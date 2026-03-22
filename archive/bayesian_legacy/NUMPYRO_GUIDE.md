# NumPyro Bayesian Implementation Guide

## Overview

The NumPyro implementation provides **full Bayesian inference** for the normative model using Markov Chain Monte Carlo (MCMC). Built on JAX for speed and GPU acceleration.

### When to Use NumPyro vs PyEM

**Use NumPyro when you want:**
- Full posterior distributions (not just point estimates)
- Uncertainty quantification for all parameters
- Credible intervals and posterior predictive checks
- Bayesian model comparison (WAIC, LOO-CV)
- To test how certain you are about parameter values

**Use PyEM when you want:**
- Fast point estimates (MLE/MAP)
- Simple parameter fitting
- Quick exploration or batch processing
- Computational efficiency over uncertainty quantification

## Installation

NumPyro requires JAX and related packages:

```bash
# CPU version
pip install numpyro jax arviz corner

# GPU version (CUDA 11)
pip install numpyro arviz corner
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Quick Start

### Basic Fitting

```python
from bayesian import run_mcmc, summarize_posterior

# Load your data
import numpy as np
states = np.load('your_data.npy')
bucket_positions = states[1]
bag_positions = states[2]

# Run MCMC (this may take a few minutes)
mcmc = run_mcmc(
    bucket_positions,
    bag_positions,
    context='changepoint',
    num_warmup=1000,      # Adaptation phase
    num_samples=2000,     # Samples per chain
    num_chains=4,         # Parallel chains
    seed=42
)

# Get posterior summary
summary = summarize_posterior(mcmc, prob=0.89)

# Print results
for param, stats in summary.items():
    print(f"{param}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"  89% HPDI: [{stats['hpdi_low']:.3f}, {stats['hpdi_high']:.3f}]")
```

### Using the Command-Line Script

```bash
# Basic usage
python scripts/fitting/fit_bayesian_numpyro.py data/your_data.npy --context changepoint

# More samples for better convergence
python scripts/fitting/fit_bayesian_numpyro.py data/your_data.npy \
    --context changepoint \
    --num-warmup 2000 \
    --num-samples 5000 \
    --num-chains 8

# Quick test run
python scripts/fitting/fit_bayesian_numpyro.py data/your_data.npy \
    --context oddball \
    --num-warmup 500 \
    --num-samples 1000 \
    --num-chains 2
```

## Understanding the Output

### MCMC Diagnostics

After running MCMC, you'll see a summary table like this:

```
                mean       std    median      5.0%     95.0%     n_eff     r_hat
    H           0.15      0.03      0.14      0.10      0.19   3247.21      1.00
    LW          0.78      0.04      0.78      0.71      0.84   3891.45      1.00
    UU          0.42      0.05      0.42      0.34      0.50   3562.89      1.00
    sigma_motor 1.23      0.11      1.22      1.05      1.41   4012.34      1.00
    sigma_LR    0.87      0.09      0.87      0.72      1.02   3789.12      1.00
```

**Key metrics:**
- **mean/median**: Central tendency of posterior
- **std**: Posterior uncertainty
- **5.0% / 95.0%**: 90% credible interval
- **n_eff**: Effective sample size (should be > 1000)
- **r_hat**: Convergence diagnostic (should be < 1.01)

### Interpreting r_hat

- **r_hat < 1.01**: ✓ Excellent convergence
- **1.01 < r_hat < 1.05**: ⚠️ Marginal, consider more samples
- **r_hat > 1.05**: ✗ Poor convergence, run longer

### Interpreting n_eff

- **n_eff > 1000**: ✓ Good effective sample size
- **n_eff < 1000**: ⚠️ Consider more samples or reparameterization
- **n_eff < 100**: ✗ Poor mixing, investigate chains

## Advanced Usage

### Posterior Predictive Checks

```python
from bayesian import posterior_predictive

# Generate predictions from posterior
predictions = posterior_predictive(
    mcmc,
    bucket_positions,
    bag_positions,
    context='changepoint',
    num_samples=1000
)

# Check predictions
observed_updates = np.diff(bucket_positions, prepend=bucket_positions[0])
predicted_updates = predictions['bucket_update']  # Shape: (1000, n_trials)

# Compute posterior predictive p-value
pred_mean = predicted_updates.mean(axis=0)
residuals = observed_updates - pred_mean
print(f"Mean absolute error: {np.abs(residuals).mean():.3f}")
```

### Model Comparison with WAIC

```python
from bayesian import run_mcmc, compute_waic

# Fit both models
mcmc_cp = run_mcmc(bucket, bag, context='changepoint', num_samples=2000)
mcmc_ob = run_mcmc(bucket, bag, context='oddball', num_samples=2000)

# Compute WAIC (lower is better)
waic_cp = compute_waic(mcmc_cp, bucket, bag, 'changepoint')
waic_ob = compute_waic(mcmc_ob, bucket, bag, 'oddball')

print(f"Changepoint WAIC: {waic_cp['waic']:.2f} ± {waic_cp['se']:.2f}")
print(f"Oddball WAIC: {waic_ob['waic']:.2f} ± {waic_ob['se']:.2f}")

# Difference > 2×SE suggests meaningful difference
diff = waic_ob['waic'] - waic_cp['waic']
se_diff = np.sqrt(waic_cp['se']**2 + waic_ob['se']**2)
if abs(diff) > 2 * se_diff:
    better = 'changepoint' if diff > 0 else 'oddball'
    print(f"Better model: {better}")
```

### Extracting Posterior Samples

```python
# Get all posterior samples
samples = mcmc.get_samples()

# samples is a dict with keys: 'H', 'LW', 'UU', 'sigma_motor', 'sigma_LR'
# Each value is a JAX array of shape (num_chains * num_samples,)

# Example: analyze H posterior
import numpy as np
H_samples = np.array(samples['H'])

print(f"H mean: {H_samples.mean():.3f}")
print(f"H median: {np.median(H_samples):.3f}")
print(f"H 95% CI: [{np.percentile(H_samples, 2.5):.3f}, "
      f"{np.percentile(H_samples, 97.5):.3f}]")

# Plot histogram
import matplotlib.pyplot as plt
plt.hist(H_samples, bins=50, density=True, alpha=0.7)
plt.xlabel('H (Hazard Rate)')
plt.ylabel('Posterior Density')
plt.title('Posterior Distribution of H')
plt.show()
```

### Custom Priors

To use different priors, modify the `normative_model` function in `bayesian/numpyro_models.py`:

```python
# Example: Informative prior based on previous study
def normative_model_informed(...):
    # More informative prior: H centered around 0.125
    H = numpyro.sample('H', dist.Beta(concentration1=2, concentration0=14))
    # This gives mean = 2/(2+14) ≈ 0.125

    # Weakly informative prior: LW likely high
    LW = numpyro.sample('LW', dist.Beta(concentration1=5, concentration0=2))
    # This gives mean = 5/(5+2) ≈ 0.71

    # ... rest of model ...
```

## Visualization

### Posterior Distributions

```python
from bayesian.visualization import plot_posterior_pairs

# Create corner plot showing all parameter correlations
fig = plot_posterior_pairs(
    mcmc,
    params=['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR'],
    save_path='figures/corner_plot.png'
)
```

### Trace Plots

```python
import arviz as az

# Convert to ArviZ InferenceData
idata = az.from_numpyro(mcmc)

# Plot traces
az.plot_trace(idata, var_names=['H', 'LW', 'UU'])
plt.tight_layout()
plt.savefig('figures/trace_plots.png', dpi=300)
```

### Energy Diagnostic

```python
from bayesian.visualization import plot_energy_diagnostic

# Check for sampling problems
fig = plot_energy_diagnostic(
    mcmc,
    save_path='figures/energy_diagnostic.png'
)

# Ideally, marginal and transition energy should overlap
# Divergence indicates sampling problems (increase num_warmup)
```

### Model Comparison

```python
from bayesian.visualization import plot_posterior_comparison

# Compare posteriors for two models
fig = plot_posterior_comparison(
    mcmc_cp,
    mcmc_ob,
    save_path='figures/posterior_comparison.png'
)
```

## Troubleshooting

### Problem: Low n_eff or high r_hat

**Solution:**
```python
# Increase warmup and samples
mcmc = run_mcmc(
    bucket, bag, context='changepoint',
    num_warmup=3000,      # More adaptation
    num_samples=5000,     # More samples
    num_chains=4
)
```

### Problem: Divergent transitions

**Symptoms:**
```
Number of divergences: 47
```

**Solutions:**
1. Increase warmup: `num_warmup=2000`
2. Use tighter priors (if you have domain knowledge)
3. Reparameterize the model (advanced)

### Problem: Very slow sampling

**Solutions:**
1. Use fewer chains initially for testing: `num_chains=2`
2. Reduce samples: `num_warmup=500, num_samples=1000`
3. Use GPU acceleration (if available)
4. Use PyEM for point estimates instead

### Problem: Import errors

**Solution:**
```bash
# Make sure all dependencies are installed
pip install numpyro jax jaxlib arviz corner

# For GPU support (optional):
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Comparison: PyEM vs NumPyro

| Feature | PyEM | NumPyro |
|---------|------|---------|
| Speed | ✓✓✓ Very fast (seconds) | ✓ Slower (minutes) |
| Output | Point estimate | Full posterior |
| Uncertainty | No | Yes (credible intervals) |
| GPU Support | No | Yes |
| Best for | Quick fits, batch processing | Detailed analysis, uncertainty |
| Parallelization | Manual | Automatic (chains) |
| Model comparison | BIC/AIC | WAIC/LOO-CV |
| Ease of use | ✓✓✓ Simple | ✓✓ Moderate |

## Example Workflow

### 1. Exploratory Analysis (PyEM)

```python
from bayesian import fit_bayesian_model
from scipy.optimize import minimize

# Quick fit to see if model makes sense
result = minimize(
    fit_bayesian_model,
    np.zeros(5),
    args=(bucket, bag, 'changepoint', None, 'nll'),
    method='Nelder-Mead'
)

# Check if fit is reasonable
print(f"Converged: {result.success}")
print(f"NegLL: {result.fun:.2f}")
```

### 2. Full Inference (NumPyro)

```python
from bayesian import run_mcmc, summarize_posterior

# Once model looks good, get full posterior
mcmc = run_mcmc(bucket, bag, 'changepoint',
               num_warmup=2000, num_samples=3000, num_chains=4)

# Check convergence
summary = summarize_posterior(mcmc)
```

### 3. Model Comparison (NumPyro)

```python
# Compare changepoint vs oddball
mcmc_cp = run_mcmc(bucket, bag, 'changepoint', num_samples=2000)
mcmc_ob = run_mcmc(bucket, bag, 'oddball', num_samples=2000)

# Use WAIC for comparison
waic_cp = compute_waic(mcmc_cp, bucket, bag, 'changepoint')
waic_ob = compute_waic(mcmc_ob, bucket, bag, 'oddball')

print(f"ΔWAIC = {waic_ob['waic'] - waic_cp['waic']:.2f}")
```

### 4. Visualization & Reporting

```python
# Generate all plots
from scripts.fitting.fit_bayesian_numpyro import (
    plot_posterior_distributions,
    plot_trace,
    plot_posterior_predictive_check
)

plot_posterior_distributions(results, save_path='figures/posterior.png')
plot_trace(results, save_path='figures/trace.png')
plot_posterior_predictive_check(results, save_path='figures/ppc.png')
```

## FAQ

**Q: How many samples do I need?**
A: Start with `num_warmup=1000, num_samples=2000, num_chains=4`. Check n_eff > 1000 and r_hat < 1.01.

**Q: Can I use GPU?**
A: Yes! If JAX detects a GPU, it will use it automatically. Expect 2-10× speedup.

**Q: How long does fitting take?**
A: With default settings (2000 samples, 4 chains), expect 2-10 minutes on CPU, < 1 minute on GPU.

**Q: What if chains don't converge?**
A: Increase `num_warmup`. If still problematic, check your data for outliers or consider using PyEM.

**Q: Can I fit multiple datasets?**
A: Yes, but run them sequentially or use process pooling (JAX doesn't play well with multiprocessing).

**Q: How do I cite NumPyro?**
A:
```
Phan, D., Pradhan, N., & Jankowiak, M. (2019).
Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro.
arXiv preprint arXiv:1912.11554.
```

## References

1. **NumPyro Documentation**: http://num.pyro.ai/
2. **JAX Documentation**: https://jax.readthedocs.io/
3. **ArviZ for Diagnostics**: https://arviz-devs.github.io/arviz/
4. **MCMC Best Practices**: Gelman et al. (2013), Bayesian Data Analysis, 3rd ed.
5. **WAIC**: Watanabe (2010), Journal of Machine Learning Research
