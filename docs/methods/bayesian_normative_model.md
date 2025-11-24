# Bayesian Normative Model: Mathematical Formulation

## Overview

This document provides the complete mathematical specification of the Bayesian normative model used for analyzing predictive inference behavior in the helicopter-bag task. The model computes optimal learning rates based on changepoint detection and uncertainty estimation.

## Model Parameters

The model has **5 free parameters**:

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Hazard rate | H | [0, 1] | Prior frequency of extreme events (changepoints) |
| Likelihood weight | LW | [0, 1] | Sensitivity to extremeness of prediction errors |
| Uncertainty underestimation | UU | [0, 1] | Degree of inappropriate uncertainty reduction |
| Motor variance | σ_motor | [0, 5] | Base variance in action execution |
| LR variance slope | σ_LR | [0, 5] | Scaling of variance with update magnitude |

## Task Structure

**Helicopter-Bag Predictive Inference Task:**
- Agent predicts bag landing position by placing a bucket
- Bag drops from helicopter with Gaussian noise: bag ~ N(helicopter, σ_N)
- Task constant: σ_N = 20
- Two contexts:
  - **Changepoint (CP)**: Helicopter location undergoes frequent shifts
  - **Oddball (OB)**: Helicopter location is stable with occasional outliers

## Core Equations

The model implements seven key equations from Loosen et al. (2023):

### Equation 1: Normative Update

The normative model prescribes updating beliefs proportional to the prediction error:

```
update_t = α_t × δ_t
```

Where:
- **update_t**: Change in bucket position from trial t-1 to trial t
- **α_t**: Learning rate (from Eq. 2 or 3)
- **δ_t**: Prediction error = bag_t - bucket_t

**Implementation:**
```python
normative_update[t] = learning_rate[t] * pred_error[t]
```

---

### Equation 2: Learning Rate (Changepoint Context)

In changepoint contexts, learning rate increases when either a changepoint is detected OR when uncertain:

```
α_t = Ω_t + τ_t - (Ω_t × τ_t)
```

Where:
- **Ω_t**: Changepoint probability (Eq. 4)
- **τ_t**: Relative uncertainty (Eq. 5)

**Interpretation:**
- High Ω_t → reset beliefs (changepoint detected)
- High τ_t → update more (uncertain about current belief)
- Interaction term prevents double-counting when both are high

**Implementation:**
```python
if context == 'changepoint':
    learning_rate[t] = omega[t] + tau[t] - (omega[t] * tau[t])
```

---

### Equation 3: Learning Rate (Oddball Context)

In oddball contexts, learning rate increases with uncertainty but *decreases* with changepoint detection:

```
α_t = τ_t - (Ω_t × τ_t)
```

**Interpretation:**
- High τ_t → update more (uncertain)
- High Ω_t → update less (oddball should be ignored, not learned from)
- Appropriate when extreme events are rare and uninformative

**Implementation:**
```python
elif context == 'oddball':
    learning_rate[t] = tau[t] - (omega[t] * tau[t])
```

---

### Equation 4: Changepoint Probability (Ω_t)

Bayesian inference over whether current observation reflects a changepoint or noise:

```
Ω_t = (H × U(δ_t)^LW) / (H × U(δ_t)^LW + (1-H) × N(δ_t)^LW)
```

Where:
- **H**: Hazard rate (prior probability of changepoint)
- **LW**: Likelihood weight (sensitivity to extremeness)
- **U(δ_t)**: Uniform likelihood = Uniform(δ_t | 0, 300)^LW
- **N(δ_t)**: Normal likelihood = N(δ_t | 0, σ_t)^LW
- **σ_t**: Current uncertainty-scaled noise = σ_N / τ_t

**Components:**

1. **Uniform likelihood** (changepoint hypothesis):
   - Extreme outcomes equally likely anywhere in range [0, 300]
   ```python
   U_val[t] = stats.uniform.pdf(pred_error[t], 0, 300) ** LW
   ```

2. **Normal likelihood** (no-changepoint hypothesis):
   - Outcomes near current belief more likely
   ```python
   sigma_t = sigma_N / tau[t]
   N_val[t] = stats.norm.pdf(pred_error[t], 0, sigma_t) ** LW
   ```

3. **Posterior odds** (via Bayes' rule):
   ```python
   omega[t] = (H * U_val[t]) / (H * U_val[t] + (1 - H) * N_val[t])
   ```

**Interpretation:**
- Large |δ_t| → U(δ_t) favored → high Ω_t (likely changepoint)
- Small |δ_t| → N(δ_t) favored → low Ω_t (likely noise)
- LW weights how much extremeness matters

---

### Equation 5: Relative Uncertainty (τ_t)

Precision-weighted integration of old belief and new observation:

```
τ_t = numerator / (numerator + σ_N)
```

Where numerator is:
```
numerator = (Ω_t × σ_N) + ((1 - Ω_t) × σ_t × τ_{t-1}) + (Ω_t × (1 - Ω_t) × [δ_t × (1 - τ_{t-1})]^2)
```

**Components:**

1. **Changepoint contribution**: Ω_t × σ_N
   - If changepoint, uncertainty increases to baseline

2. **Stable belief contribution**: (1 - Ω_t) × σ_t × τ_{t-1}
   - If no changepoint, precision improves

3. **Mixture variance**: Ω_t × (1 - Ω_t) × [δ_t × (1 - τ_{t-1})]^2
   - Uncertainty from mixing two hypotheses

**Uncertainty underestimation:**
After computing normative τ_t, apply bias parameter:
```
τ_{t+1} = τ_t / UU
```
- UU < 1 → overconfidence (uncertainty underestimated)
- UU = 1 → normative (no bias)
- UU > 1 → underconfidence (uncertainty overestimated)

**Implementation:**
```python
# First trial initialization
tau_0 = 0.5 / UU

# Trial-by-trial update
numerator = ((omega[t] * sigma_N) +
            ((1 - omega[t]) * sigma_t * tau[t]) +
            (omega[t] * (1 - omega[t]) * (pred_error[t] * (1 - tau[t]))**2))
denominator = numerator + sigma_N
this_tau = numerator / denominator
tau[t+1] = this_tau / UU
```

---

### Equation 6: Update Likelihood

Probability of observed bucket update given normative prediction:

```
L(observed_update | normative_update, σ_update) = N(observed_update | normative_update, σ_update)
```

Where:
- **observed_update**: Actual bucket movement = bucket_t - bucket_{t-1}
- **normative_update**: Model's prescribed update (Eq. 1)
- **σ_update**: Update variance (Eq. 7)

**Implementation:**
```python
bucket_update[t] = bucket_positions[t] - bucket_positions[t-1]
L_normative_update[t] = stats.norm.pdf(
    bucket_update[t],
    loc=normative_update[t],
    scale=sigma_update
)
```

**Negative log-likelihood:**
```python
negll[t] = -log(L_normative_update[t] + ε)
```
(ε = 1e-10 for numerical stability)

---

### Equation 7: Update Variance

Variance of action execution scales with update magnitude:

```
σ_update = σ_motor + |normative_update| × σ_LR
```

**Components:**

1. **σ_motor**: Base motor noise (constant variance)
2. **|normative_update| × σ_LR**: Scaled variance (larger updates → more noise)

**Implementation:**
```python
sigma_update = sigma_motor + abs(normative_update[t]) * sigma_LR
```

**Interpretation:**
- Small updates (α_t small): Variance ≈ σ_motor
- Large updates (α_t large): Variance increases proportionally
- Captures Weber's law-like scaling of motor noise

---

## Parameter Estimation

### Transformation Functions

Parameters are estimated in normalized space (-∞, ∞) then transformed to constrained ranges:

**Alpha transformation** (for H, LW, UU):
```python
def norm2alpha(x):
    """Transform x ∈ (-∞, ∞) to [0, 1]"""
    return 1 / (1 + exp(-x))
```

**Beta transformation** (for σ_motor, σ_LR):
```python
def norm2beta(x, max_val=5):
    """Transform x ∈ (-∞, ∞) to [0, max_val]"""
    return max_val / (1 + exp(-x))
```

### Optimization Objective

**Maximum Likelihood Estimation (MLE):**
```
θ_MLE = argmin_θ [ -∑_t log L(update_t | normative_update_t(θ), σ_update(θ)) ]
```

**Maximum A Posteriori (MAP):**
```
θ_MAP = argmin_θ [ -∑_t log L(update_t | θ) - log P(θ) ]
```

Where P(θ) is the prior distribution over parameters.

### Full Bayesian Inference (NumPyro)

**Priors:**
```
H ~ Beta(2, 2)
LW ~ Beta(2, 2)
UU ~ Beta(2, 2)
σ_motor ~ HalfNormal(scale)
σ_LR ~ HalfNormal(scale)
```

**Posterior sampling:**
Uses No-U-Turn Sampler (NUTS) MCMC to obtain full posterior:
```
P(θ | data) ∝ P(data | θ) × P(θ)
```

---

## Model Comparison Metrics

### Bayesian Information Criterion (BIC)

For point estimates (PyEM):
```
BIC = -2 × log L(data | θ_MLE) + k × log(n)
```

Where:
- k = 5 (number of parameters)
- n = number of trials

### Watanabe-Akaike Information Criterion (WAIC)

For Bayesian inference (NumPyro):
```
WAIC = -2 × (lppd - p_WAIC)
```

Where:
- **lppd**: Log pointwise predictive density
- **p_WAIC**: Effective number of parameters

**Lower values indicate better model fit.**

---

## Context-Specific Predictions

### Changepoint Context

**Expected behavior:**
- High learning rates after large prediction errors
- Learning rate increases with both Ω_t and τ_t
- Appropriate for volatile environments

**Equation 2 behavior:**
- α_t ≈ 1 when Ω_t = 1 (certain changepoint → full reset)
- α_t ≈ τ_t when Ω_t = 0 (no changepoint → update by uncertainty)

### Oddball Context

**Expected behavior:**
- Learning rate decreases after extreme events
- Model learns to ignore rare outliers
- Appropriate for stable environments with noise

**Equation 3 behavior:**
- α_t ≈ 0 when Ω_t = 1 (likely oddball → ignore)
- α_t ≈ τ_t when Ω_t = 0 (normal trial → update normally)

### Context Discrimination Metric

**Area between curves:**
```
Area = ∫ |α_CP(δ) - α_OB(δ)| dδ
```

Measures how well agent differentiates between changepoint and oddball contexts.
- **Higher values** → better context discrimination
- **Lower values** → context-inappropriate behavior

**Human reference values (Nassar et al. 2021):**
- Healthy controls: Area ≈ 15.3
- Schizophrenia patients: Area ≈ 10.3
- Well-performing RNNs: Area ≈ 15-30 expected

---

## Implementation Notes

### Trial-by-trial computation

The model processes data sequentially:

```python
for t in range(n_trials):
    # 1. Compute changepoint probability (Eq. 4)
    omega[t] = compute_omega(pred_error[t], H, LW, tau[t])

    # 2. Update relative uncertainty (Eq. 5)
    tau[t+1] = compute_tau(omega[t], tau[t], pred_error[t], UU)

    # 3. Compute learning rate (Eq. 2 or 3)
    learning_rate[t] = compute_alpha(omega[t], tau[t], context)

    # 4. Compute normative update (Eq. 1)
    normative_update[t] = learning_rate[t] * pred_error[t]

    # 5. Evaluate likelihood (Eq. 6, 7)
    sigma_update = compute_variance(normative_update[t], sigma_motor, sigma_LR)
    negll[t] = -log_likelihood(observed_update[t], normative_update[t], sigma_update)
```

### Numerical stability

1. **Log-likelihood computation**: Add small constant (ε = 1e-10) before log
2. **Probability bounds**: Clip values to [0, 1] after computation
3. **Variance bounds**: Ensure σ_update > 0

### Initial conditions

- **tau[0]**: Set to 0.5 / UU (moderately uncertain)
- **bucket_update[0]**: Set to bucket_positions[0] - bucket_positions[-1] (assumes wraparound)

---

## References

1. **Loosen, A. M., Skvortsova, V., & Hauser, T. U. (2023).** pyEM: A Python package for EM-based Bayesian estimation of hierarchical cognitive models. *Behavior Research Methods*. https://link.springer.com/article/10.3758/s13428-024-02427-y

2. **McGuire, J. T., Nassar, M. R., Gold, J. I., & Kable, J. W. (2014).** Functionally dissociable influences on learning rate in a dynamic environment. *Neuron*, 84(4), 870-881.

3. **Nassar, M. R., Waltz, J. A., Albrecht, M. A., Gold, J. M., & Frank, M. J. (2021).** All or nothing belief updating in patients with schizophrenia reduces precision and flexibility of beliefs. *Brain*, 144(3), 1013-1029.

---

## Code Locations

- **PyEM implementation**: `bayesian/pyem_models.py`
- **NumPyro implementation**: `bayesian/numpyro_models.py`
- **Fitting scripts**:
  - PyEM: `scripts/fitting/fit_bayesian_pyem.py`
  - NumPyro: `scripts/fitting/fit_bayesian_numpyro.py`
- **Validation**: `scripts/analysis/validate_nassar2021.py`

---

**Last Updated:** 2025-01-22
