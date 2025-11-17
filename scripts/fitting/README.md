# Bayesian Model Fitting Scripts

This directory contains scripts for fitting Bayesian normative models to behavioral data from the predictive inference task.

## Available Scripts

### 1. `fit_bayesian_pymc.py`

Fits Bayesian models using PyMC for full Bayesian inference.

**Features:**
- Maximum Likelihood Estimation (MLE)
- Model simulation
- Comparison plots

**Usage:**
```bash
# Fit model using MLE
python scripts/fitting/fit_bayesian_pymc.py data/env_data_change-point.npy \
    --model-type changepoint \
    --method mle

# Simulate data from model
python scripts/fitting/fit_bayesian_pymc.py data/env_data_change-point.npy \
    --model-type changepoint \
    --method simulate \
    --n-trials 200
```

**Arguments:**
- `data_path`: Path to behavioral data (.npy file)
- `--model-type`: Model type (changepoint or oddball)
- `--method`: Fitting method (mle or simulate)
- `--n-trials`: Number of trials for simulation
- `--no-save`: Do not save results

**Outputs:**
- Parameter estimates saved to `output/bayesian_fits/`
- Comparison plots saved to `output/behavioral_figures/`

---

### 2. `fit_bayesian_pyem.py`

Fits normative models using the PyEM framework (Loosen et al., 2023).

**Features:**
- Parameter estimation for 5 model parameters:
  - **H**: Hazard rate (frequency of extreme events)
  - **LW**: Likelihood weight (extremeness factor)
  - **UU**: Uncertainty underestimation
  - **σ_motor**: Update variance (motor noise)
  - **σ_LR**: Update variance slope
- Comprehensive model diagnostics
- Detailed visualization of fit quality

**Usage:**
```bash
# Fit PyEM model to changepoint data
python scripts/fitting/fit_bayesian_pyem.py data/env_data_change-point.npy \
    --context changepoint

# Fit to oddball data
python scripts/fitting/fit_bayesian_pyem.py data/env_data_oddball.npy \
    --context oddball

# Fit without saving results
python scripts/fitting/fit_bayesian_pyem.py data/env_data_change-point.npy \
    --context changepoint \
    --no-save \
    --no-plot
```

**Arguments:**
- `data_path`: Path to behavioral data (.npy file)
- `--context`: Task context (changepoint or oddball)
- `--no-save`: Do not save results
- `--no-plot`: Do not generate plots

**Outputs:**
- Fitted parameters saved to `output/bayesian_fits/pyem_params_{context}.npy`
- Full results saved to `output/bayesian_fits/pyem_full_results_{context}.npy`
- Diagnostic plots saved to `output/behavioral_figures/pyem_fit_{context}.png`

**Diagnostic Plots:**
1. Actual vs Predicted Positions
2. Model Learning Rate trajectory
3. Prediction Error over time
4. Changepoint Probability (Ω) over time
5. Relative Uncertainty (τ) over time
6. Actual vs Normative Update comparison

---

## Data Format

Both scripts expect behavioral data in `.npy` format with the following structure:

```python
states = np.array([
    trials,              # [0] Trial indices
    bucket_positions,    # [1] Bucket position trajectory
    bag_positions,       # [2] Bag position trajectory
    helicopter_positions,# [3] Helicopter position trajectory
    hazard_triggers      # [4] Hazard event indicators
])
```

**Shape:** `(5, n_trials)`

## Model Theory

### Normative Model Equations

1. **Learning Rate (α):**
   - Changepoint: `α = Ω + τ - Ω·τ`
   - Oddball: `α = τ - Ω·τ`

2. **Changepoint Probability (Ω):**
   ```
   Ω = (H · U(δ)) / (H · U(δ) + (1-H) · N(δ))
   ```
   where `U` is uniform distribution, `N` is normal distribution

3. **Relative Uncertainty (τ):**
   Updated based on precision-weighted integration

4. **Normative Update:**
   ```
   update = α · δ
   ```
   where `δ` is prediction error

### References

- **PyMC Approach:** Uses PyMC for Bayesian parameter estimation
- **PyEM Approach:** Based on Loosen et al. (2023)
  - Paper: https://link.springer.com/article/10.3758/s13428-024-02427-y
  - Supplement: Section 43 for detailed model equations

---

## Tips

1. **Initial Parameters:** PyEM fitting starts from default values. For difficult fits, you may need to adjust initial parameters.

2. **Convergence:** PyEM uses Nelder-Mead optimization with max 10,000 iterations. Monitor convergence messages.

3. **Data Quality:** Ensure your behavioral data has sufficient trials (recommended: >100) for stable parameter estimation.

4. **Model Comparison:** Fit both changepoint and oddball conditions to compare parameter differences.

---

## Example Workflow

```bash
# 1. Extract behavioral data (if not already done)
python scripts/data_pipeline/01_extract_model_behavior.py

# 2. Fit PyEM model to both conditions
python scripts/fitting/fit_bayesian_pyem.py \
    output/behavioral_summary/env_data_changepoint.npy \
    --context changepoint

python scripts/fitting/fit_bayesian_pyem.py \
    output/behavioral_summary/env_data_oddball.npy \
    --context oddball

# 3. Compare fitted parameters
python scripts/analysis/compare_fitted_params.py  # (if available)
```

---

## Dependencies

Required packages:
- `numpy`
- `scipy`
- `matplotlib`
- `pymc` (for PyMC fitting)
- `arviz` (for PyMC diagnostics)
- `pytensor` (for PyMC backend)
- `pyEM` (for PyEM fitting)

Install via:
```bash
pip install -e ".[fitting]"  # If fitting dependencies are optional
```

Or:
```bash
pip install numpy scipy matplotlib pymc arviz pytensor
```
