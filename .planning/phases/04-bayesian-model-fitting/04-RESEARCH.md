# Phase 4: Bayesian Model Fitting - Research

**Researched:** 2026-04-29
**Domain:** NumPyro/JAX Bayesian observer model fitting; Nassar 2021 behavioral data
**Confidence:** MEDIUM (model equations HIGH from codebase; priors LOW — paper methods unavailable without full text; .mat structure HIGH from direct inspection)

---

## Summary

Phase 4 fits the Nassar 2021 reduced Bayesian observer (RBO) model to human schizophrenia data and K=20 RNN seeds. The core model equations are already partially implemented in `src/nn4psych/bayesian/numpyro_models.py` from Phase 1, using the correct JAX-scan + `jax.lax.cond` pattern for changepoint/oddball branching. That implementation has significant gaps: it uses placeholder `Beta(2, 2)` priors rather than paper-specified priors, and the tau update equation is simplified (missing the full predictive-variance-weighted tau update from Nassar 2010). The existing code must be refactored and validated, not reused as-is.

**Critical data finding:** The two `.mat` files in `data/raw/nassar2021/` are **aggregated sliding-window regression outputs** (shape `(134, 115, 2/3)`), NOT raw trial-by-trial behavioral data. Raw trial sequences (bucket positions, bag positions, per-trial context labels) live in a separate `Nassar_et_al_2021/Brain2021Code/realSubjects/` directory that is **not present on this machine**. That data is downloadable from the Nassar lab resources page (Google Drive link confirmed via WebFetch). Plan 04-03 must include a data-fetch sub-task before any fitting code runs.

**Primary recommendation:** Implement the RBO forward model from scratch in a new `src/nn4psych/bayesian/reduced_bayesian.py`, reusing the proven `jax.lax.cond` + `jnp.bool_` + `jax.lax.scan` skeleton from `numpyro_models.py` lines 108–162 but correcting the tau update equation and replacing placeholder priors. Run ArviZ diagnostics via `az.from_numpyro` + `az.rhat` / `az.ess`.

---

## Standard Stack

### Core (installed in `actinf-py-scripts` env, confirmed)

| Library | Installed Version | Purpose | Source |
|---------|-------------------|---------|--------|
| jax | 0.4.35 | JIT-compiled forward model, `jax.lax.scan`, `jax.lax.cond` | `pip show jax` |
| numpyro | 0.19.0 | NUTS MCMC, `Predictive`, `MCMC`, `NUTS` | `pip show numpyro` |
| arviz | 0.23.4 | `az.from_numpyro`, `az.rhat`, `az.ess`, `az.summary` | installed 2026-04-29 |
| scipy | existing | `.mat` file loading via `scipy.io.loadmat` | existing |
| numpy | existing | Array manipulation | existing |

**Note:** `pyproject.toml` lists `arviz>=0.17.0` in the `[bayesian]` extra. The installed version is 0.23.4. The ArviZ 0.23.x API emits a FutureWarning about upcoming refactoring but all used APIs (`from_numpyro`, `rhat`, `ess`, `summary`) are present and functional. Use `warnings.filterwarnings('ignore', category=FutureWarning)` in scripts.

**Installation (if re-creating env):**
```bash
pip install "jax>=0.4.35" "numpyro>=0.19.0" "arviz>=0.17.0"
```

### Supporting

| Library | Purpose | When to Use |
|---------|---------|-------------|
| scipy.io | Load `.mat` files (`loadmat` with `squeeze_me=True, struct_as_record=False`) | 04-03 data loading |
| matplotlib (Agg) | Recovery scatter plots, trace plots | 04-02 |
| pandas | Per-fit summary tables | 04-03, 04-04 |
| json | Trimmed posterior JSON output | every fit |

---

## Section 1: Nassar 2021 Reduced Bayesian Observer — Model Spec

### State equations (confirmed from `src/nn4psych/utils/metrics.py` lines 196–281 and `src/nn4psych/bayesian/numpyro_models.py`)

The reduced Bayesian observer tracks three latent quantities per trial:

**Changepoint/oddball probability (Ω_t)** — same for both conditions:
```
U_val = Uniform(delta_t; 0, 300) ^ LW
N_val = Normal(delta_t; 0, sigma_t) ^ LW    # sigma_t = sigma_N / tau_{t-1}
Omega_t = (H * U_val) / (H * U_val + (1 - H) * N_val + eps)
```
where `delta_t = bag_t - bucket_t` (prediction error), `sigma_N = 20.0` (bag placement noise, confirmed from Nassar 2021 PMC text), `LW` is the likelihood weight exponent that scales how much the extremeness of an outcome drives CP/OB detection.

**Relative uncertainty (tau_t) update:**

The current implementation in `numpyro_models.py` lines 133–138 computes tau via a weighted numerator/denominator:
```python
numerator = (omega_t * sigma_N) + ((1 - omega_t) * sigma_t * tau_prev) + \
            (omega_t * (1 - omega_t) * (delta * (1 - tau_prev))**2)
denominator = numerator + sigma_N
this_tau = numerator / denominator
tau_next = this_tau / UU   # uncertainty underestimation
```
This is the existing project-internal implementation (Equations 4–5 from Loosen et al. 2023 / McGuire et al. 2014). Confidence: MEDIUM — this is what's been running but may not exactly match Nassar 2021 supplement. The planner should flag this for 04-02 validation against known parameter recovery benchmarks.

The simplified version documented in `src/nn4psych/utils/metrics.py` (line 278) is `tau_next = tau / UU` — this is the collapsed per-trial step, not the full predictive-variance calculation. Use the full version from `numpyro_models.py`.

**Learning rate (alpha_t):**
```
# Changepoint condition (Eq. 2):
alpha_t = Omega_t + tau_{t-1} - (Omega_t * tau_{t-1})

# Oddball condition (Eq. 3):
alpha_t = tau_{t-1} - (tau_{t-1} * Omega_t)
```
Confirmed from `src/nn4psych/utils/metrics.py` lines 196–235 and `numpyro_models.py` lines 150–155.

**Mean estimate update:**
```
mu_t = mu_{t-1} + alpha_t * delta_t
```

**Observation likelihood (Equations 6–7):**
```
bucket_update_t ~ Normal(alpha_t * delta_t,  sigma_motor + |alpha_t * delta_t| * sigma_LR)
```
This uses the _update_ (diff of bucket positions) as the observable, not the raw bucket position. Confirmed from `numpyro_models.py` lines 251–267.

### Free parameters

| Parameter | Symbol | Range | Role | Current Prior in Code |
|-----------|--------|-------|------|-----------------------|
| Hazard rate | H | (0, 1) | Prior probability of CP or OB event per trial | `Beta(2, 2)` — placeholder |
| Likelihood weight | LW | (0, 1) or > 0 | Exponent scaling how extreme outcomes drive Ω | `Beta(2, 2)` — placeholder |
| Uncertainty underestimation | UU | > 0 | Scales how fast tau shrinks (< 1 = underestimate) | `Beta(2, 2)` — placeholder |
| Motor noise | sigma_motor | > 0 | Baseline update variance | `HalfNormal(1.0)` — placeholder |
| Update variance slope | sigma_LR | > 0 | Update variance scaling with update magnitude | `HalfNormal(1.0)` — placeholder |

**Paper-specified priors:** The Nassar 2021 PMC full text states that "parameter estimates were regularized by refitting using posterior probability maximization and an informed prior over parameters derived from the original maximum likelihood fits." The exact distributional form of the priors is in the supplementary methods, which is not available without downloading the full supplement (available via the Google Drive link on sites.brown.edu/mattlab/resources/). Confidence: LOW.

**Recommended weakly-informative defaults** (if paper priors cannot be extracted before planning):
```python
H          ~ Beta(1.5, 8)        # Favors low hazard rates (true rate = 0.125)
LW         ~ Beta(2, 2)          # Symmetric, centered at 0.5 (can be > 1 for full model)
UU         ~ HalfNormal(0.5)     # Small positive, near 1.0 = no underestimation
sigma_motor ~ HalfNormal(10.0)   # In bag-position units (range 0–300)
sigma_LR   ~ HalfNormal(1.0)     # Multiplicative slope, positive
```
The planner must flag this as a MUST-VERIFY item: fetch the Brain2021Code supplement before 04-01 ships. The researcher confirms: exact prior extraction from Nassar 2021 supplement is required.

### JAX-traceability notes

The existing pattern in `numpyro_models.py` lines 108–162 is the canonical solution for this phase:

```python
# CORRECT PATTERN (from numpyro_models.py, confirmed working from Phase 1):
is_changepoint = jnp.bool_(context == 'changepoint')   # outside step_fn

def step_fn(carry, t):
    tau_prev = carry
    delta = pred_errors[t]
    # ... compute omega_t, tau_next ...
    lr_t = jax.lax.cond(
        is_changepoint,
        lambda _: omega_t + tau_prev - (omega_t * tau_prev),   # CP eq
        lambda _: tau_prev - (omega_t * tau_prev),              # OB eq
        operand=None,
    )
    return tau_next, (lr_t, omega_t, tau_next)

_, outputs = jax.lax.scan(step_fn, tau_0, jnp.arange(n_trials))
```

**Key constraints:**
- `context` must be resolved to `jnp.bool_` BEFORE `step_fn` is defined (Phase 1 lesson from 01-03)
- `jax.lax.cond` with `operand=None` and `lambda _:` is the required pattern — closed-over traced variables are fine
- Do NOT use Python `if/else` inside `jax.lax.scan` — silently produces dead code
- `sigma_N = 20.0` is a fixed constant (from paper), not a sampled parameter

---

## Section 2: Existing NumPyro/JAX Infrastructure

### Files present after Phase 1

**`src/nn4psych/bayesian/__init__.py`** — exports:
- `run_mcmc(bucket_positions, bag_positions, context, num_warmup=1000, num_samples=2000, num_chains=4, seed=42)` → returns `MCMC` object
- `summarize_posterior(mcmc, prob=0.89)` → dict of {mean, std, median, hpdi_low, hpdi_high} per param
- `posterior_predictive(mcmc, bucket_positions, bag_positions, context)` → dict of predictive samples
- `compute_waic(mcmc, bucket_positions, bag_positions, context)` → {waic, p_waic, se}
- `get_map_estimate(mcmc)` → dict of posterior means

**`src/nn4psych/bayesian/numpyro_models.py`** — core implementation:
- `compute_normative_model(params, pred_errors, context, sigma_N=20.0)` → `(learning_rate, normative_update, omega, tau)` using `jax.lax.scan`
- `normative_model(bucket_positions, bag_positions, context, prior_scale=1.0)` → NumPyro model function with MCMC-compatible priors and plate
- The JAX-tracer-compatible `jax.lax.cond` + `jnp.bool_` pattern is fully implemented and tested in Phase 1

**`src/nn4psych/bayesian/model_comparison.py`** — BIC/AIC/likelihood ratio (PyEM-era, kept for Phase 5 WAIC comparison)

**`src/nn4psych/bayesian/visualization.py`** — existing plot utilities (check before reimplementing)

### Gaps for Phase 4

1. **Priors**: Current `normative_model` uses `Beta(2, 2)` for H, LW, UU and `HalfNormal(1.0)` for sigma_motor, sigma_LR. These are acknowledged placeholders (docstring says "weakly informative"). Replace with paper-specified priors in 04-01.

2. **MCMC configuration**: `run_mcmc` defaults to `num_warmup=1000, num_samples=2000`. Phase 4 requires `num_warmup=2000, num_samples=2000, target_accept_prob=0.95`. The `NUTS` constructor in the current code does not pass `target_accept_prob`. Add it.

3. **ArviZ diagnostics**: `summarize_posterior` uses `numpyro.diagnostics.hpdi` but does NOT compute R-hat or ESS. Add `az.from_numpyro`, `az.rhat`, `az.ess` calls.

4. **Divergence extraction**: Not implemented. Add `extra_fields=("diverging",)` to `mcmc.run()`.

5. **Multi-CPU chains**: 4 chains on CPU requires `XLA_FLAGS=--xla_force_host_platform_device_count=4` set BEFORE `import jax`. Set in `__init__.py` or at script top. Confirmed working: setting this env var before `import jax` gives `jax.local_device_count() == 4`.

6. **PyEM archive**: `archive/bayesian_legacy/pyem_models.py` and `archive/bayesian_pymc/` exist. `cluster/batch_fit_bayesian.py` has a TODO comment referencing `fit_bayesian_model` (removed). BAYES-01 is not yet complete — the archive exists but `batch_fit_bayesian.py` still imports from removed functions.

---

## Section 3: Nassar .mat Data Structure

### What was directly inspected (HIGH confidence)

The two files in `data/raw/nassar2021/` are **sliding-window regression output files**, not raw trial data:

```
slidingWindowFits_subjects_23-Nov-2021.mat
  └── binRegData (MatlabStruct)
        ├── subRunCoeffsOdd:  ndarray (134, 115, 2)  float64
        └── subRunCoeffsCP:   ndarray (134, 115, 2)  float64

slidingWindowFits_model_23-Nov-2021.mat
  └── binRegData (MatlabStruct)
        ├── subRunCoeffsOdd:  ndarray (134, 115, 3)  float64
        └── subRunCoeffsCP:   ndarray (134, 115, 3)  float64
```

**Dimension interpretation:**
- Axis 0 (134): subjects — all 134 show non-NaN data (94 patients + ~40 controls, consistent with "108 patients + 33 controls enrolled minus exclusions = 134 total with data files")
- Axis 1 (115): sliding window positions along the sorted-PE trial sequence
- Axis 2 (2): regression coefficients [intercept, slope] for subjects; (3): three model outputs for model file
- These are NOT raw bucket/bag positions. They are derived regression statistics from Nassar 2021 Figure 2's binned-regression analysis of learning rate vs PE magnitude.

**Value ranges observed:**
- `subRunCoeffsOdd[:,:,0]` (intercept): min=-462, max=853, mean=-0.71
- `subRunCoeffsOdd[:,:,1]` (slope): min=-5.3, max=7.3, mean=0.34

### What is NOT present (confirmed missing)

The raw per-trial behavioral data (bag positions, bucket positions, trial-by-trial updates, context labels) is in a separate repository:
- Path referenced by `extract_nassar_trials.py` (line 36): `C:/Users/aman0087/Documents/Github/Nassar_et_al_2021/Brain2021Code/realSubjects/`
- This directory does NOT exist on this machine.
- Structure inferred from `extract_nassar_trials.py`: per-subject MATLAB files with `statusData` (100 trial blocks) containing `currentOutcome`, `currentPrediction`, `currentUpdate`, `currentDelta`, `blockCompletedTrials`, `currentMean`, `isChangeTrial`.
- 4 conditions: `cloud_cp_avoid`, `cloud_cp_seek`, `cloud_drift_avoid`, `cloud_drift_seek`
- 100 trials per condition block; data cleaning drops first 3 trials per block

### Data acquisition plan (prerequisite for 04-03)

The Nassar lab publishes code + data for this paper at:
- **Lab resources page:** https://sites.brown.edu/mattlab/resources/
- **Direct download:** Google Drive link (confirmed via WebFetch)

**Plan 04-03 MUST include as first task:** Download `Brain2021Code` zip, place in `data/raw/nassar2021/Brain2021Code/`, verify `realSubjects/` structure matches what `extract_nassar_trials.py` expects.

**Note on generative parameters:** The `statusData` struct includes `currentMean` (helicopter position = generative mean) and `isChangeTrial`. These ARE present in the raw files. Parameter recovery can therefore use fresh prior-sampled parameters (not the ground truth from the human task, since the human task's generative parameters don't correspond to the Bayesian observer's internal parameters H, LW, UU).

---

## Section 4: Parameter Recovery Design Specifics

### Sampling from NumPyro priors (BAYES-06)

Use `Predictive` with `num_samples` and no `posterior_samples` argument to draw prior samples:

```python
# Source: NumPyro Predictive API, confirmed working with numpyro 0.19.0
from numpyro.infer import Predictive
import jax

# Sample 50 parameter sets from the prior
prior_sampler = Predictive(reduced_bayesian_model, num_samples=50)
rng_key = jax.random.PRNGKey(0)
prior_samples = prior_sampler(rng_key)
# prior_samples['H'].shape == (50,)
# prior_samples['LW'].shape == (50,)
# prior_samples['UU'].shape == (50,)
```

This is the correct pattern — no `posterior_samples` argument means sampling from the prior.

### Synthetic trial generation

For each of the 50 sampled parameter sets:
1. Generate a trial sequence matching the Nassar paradigm:
   - CP condition: 100 trials, hazard = 0.125, bag ~ N(heli, 20), heli jumps with prob H_true
   - OB condition: 100 trials, hazard = 0.125, drift SD = 7.5, occasional bags from Uniform(0, 300)
2. Run the participant forward model: given the generative sequence, simulate bucket updates as `Normal(alpha_t * delta_t, sigma_motor + |alpha_t * delta_t| * sigma_LR)` with the sampled parameters
3. Fit MCMC to the synthetic (bag_positions, bucket_positions) pair
4. Compare recovered posterior mean to true sampled parameter

**Key constraint from CONTEXT.md:** Use Nassar paradigm trial sequences (not arbitrary sequences). The recovery validates the actual fitting pipeline used on human data.

### ArviZ recovery report

```python
# Source: ArviZ 0.23.4 confirmed API
import arviz as az
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

idata = az.from_numpyro(mcmc)
rhat_vals = az.rhat(idata)           # per-param R-hat (xarray Dataset)
ess_vals = az.ess(idata, method='bulk')  # per-param ESS_bulk

# Check gates
rhat_max = max(float(rhat_vals[v]) for v in rhat_vals.data_vars)
ess_min = min(float(ess_vals[v]) for v in ess_vals.data_vars)
passes = (rhat_max <= 1.01) and (ess_min >= 400)

# Summary table
summary_df = az.summary(idata, var_names=['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR'])
```

Recovery correlation is computed over the 50 synthetic fits: `scipy.stats.pearsonr(true_params[:, i], posterior_means[:, i])` for each parameter i. All parameters must satisfy r >= 0.85.

---

## Section 5: MCMC + Retry Implementation Pattern

### Canonical NUTS configuration (from CONTEXT.md)

```python
# Source: NumPyro 0.19.0 NUTS API (confirmed by inspecting NUTS.__init__ signature)
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
import jax
import numpyro
from numpyro.infer import MCMC, NUTS

kernel = NUTS(
    model_fn,
    target_accept_prob=0.95,   # parameter name confirmed: target_accept_prob
    max_tree_depth=10,         # conservative default
)

mcmc = MCMC(
    kernel,
    num_warmup=2000,
    num_samples=2000,
    num_chains=4,
    progress_bar=False,        # set True for interactive debugging
)
mcmc.run(
    jax.random.PRNGKey(seed),
    *model_args,
    extra_fields=('diverging',),   # REQUIRED to extract divergence count
)
```

**Critical:** `XLA_FLAGS` must be set before `import jax`. In scripts, do this at the very top before all other imports. In `src/nn4psych/bayesian/__init__.py`, `os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")` is already set — add the XLA_FLAGS line there as well.

### Diagnostics extraction

```python
import arviz as az
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

idata = az.from_numpyro(mcmc)
rhat_vals = az.rhat(idata)
ess_vals = az.ess(idata, method='bulk')

rhat_max = max(float(rhat_vals[v]) for v in rhat_vals.data_vars)
ess_min = min(float(ess_vals[v]) for v in ess_vals.data_vars)

div_field = mcmc.get_extra_fields().get('diverging', None)
n_divergences = int(div_field.sum()) if div_field is not None else 0

passes_rhat = rhat_max <= 1.01
passes_ess = ess_min >= 400
```

### Retry loop pattern

```python
def fit_with_retry(model_fn, model_args, seed):
    """Run MCMC with auto-retry on R-hat/ESS gate failure."""
    attempts = []
    for attempt_idx, (n_warmup, target_accept) in enumerate([
        (2000, 0.95),   # first attempt: standard
        (4000, 0.99),   # retry: 2x warmup, higher target_accept
    ]):
        mcmc = MCMC(
            NUTS(model_fn, target_accept_prob=target_accept, max_tree_depth=10),
            num_warmup=n_warmup,
            num_samples=2000,
            num_chains=4,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(seed + attempt_idx), *model_args,
                 extra_fields=('diverging',))

        idata = az.from_numpyro(mcmc)
        rhat_max = max(float(az.rhat(idata)[v]) for v in az.rhat(idata).data_vars)
        ess_min = min(float(az.ess(idata, method='bulk')[v])
                      for v in az.ess(idata, method='bulk').data_vars)
        n_div = int(mcmc.get_extra_fields()['diverging'].sum())

        attempt_record = {
            'attempt': attempt_idx + 1,
            'num_warmup': n_warmup,
            'target_accept': target_accept,
            'rhat_max': rhat_max,
            'ess_min': ess_min,
            'n_divergences': n_div,
            'passed_rhat': rhat_max <= 1.01,
            'passed_ess': ess_min >= 400,
        }
        attempts.append(attempt_record)

        if attempt_record['passed_rhat'] and attempt_record['passed_ess']:
            return mcmc, 'PASS', attempts

    return mcmc, 'FAILED', attempts  # both attempts failed; mark in JSON, continue
```

### Trimmed JSON output per fit

```python
def make_fit_summary(mcmc, subject_id, condition, status, attempts):
    """Produce lightweight per-fit JSON (~10 KB)."""
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    idata = az.from_numpyro(mcmc)
    summary = az.summary(idata, var_names=['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR'])

    param_stats = {}
    for param in ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR']:
        row = summary.loc[param]
        hpdi = az.hdi(idata, hdi_prob=0.95, var_names=[param])
        param_stats[param] = {
            'mean': float(row['mean']),
            'median': float(row['median']),
            'sd': float(row['sd']),
            'hdi_2.5': float(hpdi[param].values[0]),
            'hdi_97.5': float(hpdi[param].values[1]),
            'rhat': float(row['r_hat']),
            'ess_bulk': float(row['ess_bulk']),
        }

    return {
        'subject_id': subject_id,
        'condition': condition,
        'status': status,
        'params': param_stats,
        'attempts': attempts,
    }
```

---

## Section 6: RNN Cohort Logistics (04-04)

### Re-training K=20 RNN seeds

The canonical RNN is `scripts/training/train_rnn_canonical.py` (PIE_CP_OB_v2 environment). For 04-04, K=20 independent seeds of this RNN are needed.

**Cluster template:** `cluster/run_circuit_ensemble.sh` — the latent circuit ensemble pattern. For RNN re-training, a simpler SLURM script is needed (no ensemble fitting, just training loop × 20 seeds).

**Phase 3 timing reference:**
- 03-02: 100-init ensemble fitting took 204.5 min GPU (cluster job)
- Context-DM training: ~150 epochs × 200 trials = ~50 min on GPU (from `run_circuit_ensemble.sh` comment)
- `train_rnn_canonical.py`: PIE task, smaller; estimate ~30–60 min per seed on GPU

**Recommendation for 04-04:** Use a SLURM array job (not a sequential loop):
```bash
#SBATCH --array=0-19  # K=20 seeds
#SBATCH --gres=gpu:1
python scripts/training/train_rnn_canonical.py --seed ${SLURM_ARRAY_TASK_ID} --epochs 150 --trials 200
```
Each array job takes ~30–60 min. Total wall time ≈ 60 min (all parallel). Memory: ~300–400 MB per job (from project CLAUDE.md note on local memory).

**Important:** This is distinct from the latent circuit fitting. Phase 4 trains new seeds of the canonical PIE RNN (not the context-DM RNN from Phase 3). The model class is `ActorCritic` from `nn4psych.models.actor_critic`.

### Replay-human-sequences workflow

For each RNN seed × human subject:
1. Load human trial sequence: bag positions + context label (CP or OB) for ~100 trials per condition
2. Drive the RNN on this exact sequence (the RNN chooses bucket positions given bag observations)
3. Extract the resulting (bucket_positions, bag_positions) pairs
4. Fit the Reduced Bayesian model to this RNN behavior

This matches the matched-stimulus approach from Daw lab Two-Step replay studies. The RNN sees the same generative process as humans but produces different bucket trajectories. Fitting the same model to both allows direct parameter comparison.

**Per-seed output:** One JSON fit file per (seed, subject, condition). With K=20 seeds × 134 subjects × 2 conditions = 5360 fits. At ~10 KB each = ~54 MB total.

**Pooling across modality_context:** The canonical PIE RNN has `modality_context` as a feature. Pool across both modality_contexts (0 and 1) per subject replay — average or concatenate the CP and OB trials before fitting, consistent with CONTEXT.md decision.

### Compute budget estimate

| Task | Wall time | Memory | Notes |
|------|-----------|--------|-------|
| Re-train 1 RNN seed (GPU, cluster) | ~30–60 min | ~400 MB GPU | SLURM array × 20 |
| Single MCMC fit (CPU, 4 chains, 2k+2k) | ~2–10 min | ~200 MB | Depends on trial count |
| Human fits: 134 × 2 × 2 retries | ~1–5 hrs total | — | Embarrassingly parallel |
| RNN fits: 20 × 134 × 2 | ~5–20 hrs total | — | Cluster batch |

For RNN fits, a SLURM array over seeds (20 jobs) that runs all 134 subjects per seed sequentially is the simplest pattern. Estimated: 20 array jobs × ~6 hrs each = 6 hrs wall time.

---

## Section 7: CHMM / CRP Variants for 04-05

### Survey of candidate models

| Model | Reference | Context inference | JAX vectorizability | Complexity |
|-------|-----------|-------------------|---------------------|------------|
| Anderson local-MAP | Anderson (1991) + Collins & Frank (2013) | CRP prior, MAP updates each trial | MEDIUM: DP clusters need dynamic allocation | High |
| Gershman-Niv non-parametric | Gershman & Niv (2010, 2012) | Full CRP posterior | LOW: particle filter or sequential MAP | High |
| Sanborn particle filter | Sanborn et al. (2010) | Sequential Monte Carlo with CRP | LOW: particles require loop-based update, not scan-friendly | High |
| Fixed-K HMM | Any HMM textbook | Pre-specified K hidden states, Viterbi/forward-backward | HIGH: `jax.lax.scan` directly applicable | Low |
| 2-state CP/OB context HMM | Custom (simplest) | K=2 states (CP regime, OB regime), learned transition probs | HIGH: exact and vectorizable | Very Low |

**Recommendation for 04-05:** Use the **2-state contextual HMM** (K=2 fixed: CP state and OB state). This is the simplest model that captures context inference for the CP+OB task:
- State transitions: Dirichlet-distributed row stochastic matrix (prior: symmetric Dirichlet)
- Emission model per state: Reduced Bayesian observer equations (state-specific H and LW)
- Inference: forward-backward algorithm via `jax.lax.scan` over the trial sequence
- Implementation: HMM forward-backward in JAX is well-documented and scan-compatible

**Why not CRP for 04-05:** The CRP requires dynamic cluster allocation during the forward pass. This breaks `jax.lax.scan` because the state space grows with observations. A particle filter workaround exists but requires a fixed max particles — adding significant complexity for a prototype. For a gated single-subject prototype, the 2-state HMM delivers the key insight (can the model infer CP vs OB context?) with much less code risk.

**Reference for HMM in JAX/NumPyro:**
```python
# NumPyro has built-in HMM support:
from numpyro.contrib.control_flow import scan
# Or use distrax (JAX-native HMM)
# Simplest approach: implement forward-backward manually using jax.lax.scan
```

---

## Section 8: Pitfalls + Risks

### Pitfall 1: Weak identifiability of H and LW

**What goes wrong:** H (hazard rate) and LW (likelihood weight) are jointly determined by extreme outcomes. If a subject rarely makes large errors, Omega_t ≈ 0 throughout, and neither H nor LW is identifiable from the data (both produce the same Omega_t ≈ 0).

**Why it happens:** LW exponentiates both U_val and N_val, so for very small delta_t, both are nearly equal regardless of LW. H is not identifiable when Omega is always near-zero because H * U_val is the numerator — small U_val suppresses the effect of H.

**How to avoid:** Prior recovery check (04-02) will reveal this — low r for H or LW is the warning sign. If recovery fails, consider fixing H = 0.125 (true value) and treating LW and UU as the primary fit parameters, or add a Beta prior that regularizes toward the ground truth hazard rate.

**Warning signs:** R-hat inflated for H or LW specifically; bimodal posteriors visible in trace plots.

### Pitfall 2: JAX tracing inside scan — context branching

**What goes wrong:** If `context` is a Python string passed as a model argument, `jax.lax.scan` will silently trace only one branch. This is the bug fixed in Phase 1 (01-03).

**Prevention:** Always compute `is_changepoint = jnp.bool_(context == 'changepoint')` OUTSIDE the `step_fn`, as a closed-over constant. Do not pass `context` as a scan carry or argument.

**Reference:** `numpyro_models.py` lines 108–109 — the correct pattern is there already.

### Pitfall 3: 4 chains on CPU without virtual devices

**What goes wrong:** NumPyro emits `UserWarning: There are not enough devices to run parallel chains: expected 4 but got 1`. Chains run sequentially, wall time is 4× slower.

**Prevention:** Set `XLA_FLAGS='--xla_force_host_platform_device_count=4'` BEFORE any `import jax` statement. Do NOT call `numpyro.set_host_device_count(4)` after jax is imported — it has no effect. Confirmed pattern: set the env var at module top or in the script's `__main__` block before imports.

### Pitfall 4: .mat file scipy loading edge cases

**What goes wrong:** `scipy.io.loadmat` with `squeeze_me=False` returns nested arrays of `mat_struct` objects that require `.flat[0]` unpacking. `squeeze_me=True, struct_as_record=False` simplifies this but may still require `getattr(struct, field)`.

**Prevention:** Use `scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)`. Directly confirmed working: the `binRegData` struct was successfully accessed as `mat['binRegData'].subRunCoeffsOdd`. For the per-subject raw files (once obtained), use the pattern from `extract_nassar_trials.py` (lines 91–113): `np.asarray(block['field']).flatten()[0]` for scalar extraction from struct arrays.

### Pitfall 5: ArviZ 0.23.x FutureWarning noise

**What goes wrong:** ArviZ 0.23.4 emits a `FutureWarning` on import about upcoming API refactoring. This floods log output.

**Prevention:** Add to all scripts using ArviZ:
```python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='arviz')
```

### Pitfall 6: Missing raw behavioral data for 04-03

**What goes wrong:** `data/raw/nassar2021/` contains only `slidingWindowFits_*.mat` (aggregated statistics). The raw per-trial behavioral data needed to run the forward model and compute likelihood does not exist on this machine.

**Prevention:** Plan 04-03 task 1 must fetch the Brain2021Code data from the Nassar lab resources page (sites.brown.edu/mattlab/resources/ → Google Drive link). This is a MUST-COMPLETE prerequisite before any fitting code in 04-03 can run. Document the expected directory structure from `extract_nassar_trials.py`.

### Pitfall 7: InferenceData memory accumulation

**What goes wrong:** Creating `az.from_numpyro(mcmc)` for 5000+ fits would accumulate large xarray objects if kept in memory.

**Prevention:** Extract diagnostics from `az.from_numpyro(mcmc)` immediately, write the trimmed JSON, then delete the `idata` object. Do not store `MCMC` objects. Already mitigated by the trimmed-JSON-only decision in CONTEXT.md.

### Pitfall 8: sigma_N units

**What goes wrong:** The bag placement noise `sigma_N = 20.0` is in screen units (0–300 range). If the data loader normalizes bag/bucket positions to [0, 1], the Gaussian likelihood becomes nonsensical.

**Prevention:** Do NOT normalize positions. Keep raw screen-space coordinates (0–300). The likelihood `Normal(delta; 0, sigma_N / tau_prev)` only makes sense when delta is in the same units as sigma_N.

---

## Section 9: Reference Implementations to Validate Against

### No existing NumPyro/JAX port found

A WebSearch for NumPyro/Stan/PyMC ports of the Nassar reduced Bayesian observer returned no specific results. No public reference implementation in NumPyro or JAX was identified.

### Existing implementations in the project (MEDIUM confidence)

The project itself contains two partial implementations:
- `src/nn4psych/bayesian/numpyro_models.py` — current main implementation (NumPyro + JAX scan)
- `archive/bayesian_legacy/bayesian_models.py` — legacy PyEM/scipy optimization
- `archive/bayesian_pymc/pyem_models.py` — MAP estimation via `scipy.optimize.minimize`

The MAP estimates from the archive can serve as a validation check: after MCMC fitting, the posterior mean should approximately match what `scipy.optimize.minimize` produces (for well-identified parameters in simple synthetic cases).

### Validation strategy for 04-02

1. Generate a synthetic dataset with known ground-truth parameters (H=0.125, LW=0.7, UU=0.9, sigma_motor=5.0, sigma_LR=0.3)
2. Fit with NumPyro NUTS
3. Check that posterior mean recovers these values within ± 1 SD
4. Cross-check against archive MAP fit (should give similar point estimates)
5. Run full 50-dataset recovery (BAYES-06)

---

## Section 10: Plan Structure Recommendation

### Confirmed 4-plan sketch + gated 04-05

The CONTEXT.md plan sketch (04-01 through 04-04 + gated 04-05) is sound. Suggested refinements:

**04-01: Archive + Implement (Wave 1)**
- Task 1: Verify `cluster/batch_fit_bayesian.py` import of removed `fit_bayesian_model` does not break; add docstring noting it's obsolete (BAYES-01 completion)
- Task 2: Create `src/nn4psych/bayesian/reduced_bayesian.py` with corrected forward model, paper-matching priors, and Phase 1's JAX-scan pattern
- Task 3: Update `src/nn4psych/bayesian/__init__.py` to export from `reduced_bayesian.py`
- Task 4: Update `run_mcmc` defaults in new module (num_warmup=2000, target_accept_prob=0.95, extra_fields=('diverging',))
- Task 5: Add `XLA_FLAGS` env var to `__init__.py` for 4-CPU-chain support
- Estimated: 1 day

**04-02: Diagnostics + Parameter Recovery (Wave 2)**
- Task 1: Implement `run_diagnostics(mcmc)` → dict with rhat_max, ess_min, n_divergences
- Task 2: Implement retry loop `fit_with_retry()`
- Task 3: Implement `make_fit_summary()` → trimmed JSON
- Task 4: Generate 50 synthetic Nassar-paradigm trial sequences
- Task 5: Run 50 MCMC fits (locally, using small test with `--smoke_test` flag first)
- Task 6: Compute per-parameter recovery correlations; generate scatter plots
- Task 7: Update REQUIREMENTS.md SC-3 wording on divergences (document, not gate)
- Estimated: 2 days

**04-03: Fit Human Data (Wave 3)**
- Task 1: **Download Brain2021Code data** from Nassar lab (prerequisite, blocks all other tasks)
- Task 2: Wire `scripts/data_pipeline/09_fit_human_data.py`
- Task 3: Iterate over (subject, condition) cells, call `fit_with_retry`, save JSON
- Task 4: Aggregate all per-subject JSONs into a summary CSV
- Task 5: Validate against `slidingWindowFits_subjects_23-Nov-2021.mat` (model-predicted curves should match empirical curves qualitatively)
- Estimated: 1 day coding + time to download data

**04-04: Fit RNN Cohort (Wave 4 — split into two sub-plans)**

Consider splitting 04-04 into:
- **04-04a: Re-train K=20 RNN seeds** (cluster, SLURM array job, ~1 hr wall time)
- **04-04b: Replay + Fit** (cluster, SLURM array over seeds, ~6 hrs wall time)

The split is motivated by compute dependency: 04-04b cannot start until 04-04a completes and checkpoints are pulled from cluster. If the planner keeps them in one plan, add explicit dependency gates.

- Task 04-04a-1: Write `cluster/run_rnn_cohort.sh` (SLURM array, K=20 seeds)
- Task 04-04a-2: Pull trained checkpoints from cluster to `data/processed/rnn_cohort/`
- Task 04-04b-1: Write replay logic: load human trial sequence → drive RNN → extract (bucket, bag) pairs
- Task 04-04b-2: Wire `scripts/data_pipeline/10_fit_rnn_data.py`
- Task 04-04b-3: SLURM array over seeds (20 jobs × 134 subjects × 2 conditions)
- Task 04-04b-4: Aggregate per-seed JSONs
- Estimated: 2 days coding + cluster time

**04-05: CHMM Prototype (gated, Wave 5)**
- Author only after 04-04 ships
- Implement 2-state contextual HMM (K=2) using `jax.lax.scan` forward-backward
- Fit to single human subject, single condition
- Posterior predictive check vs empirical learning rate curve
- Estimated: 1–2 days if implementing, defer otherwise

### Dependencies

```
04-01 → 04-02 → 04-03 (blocked on data download)
04-01 → 04-04a → 04-04b
04-04 → 04-05 (gated, can defer)
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| R-hat calculation | Custom Gelman-Rubin | `az.rhat(az.from_numpyro(mcmc))` | ArviZ implements rank-normalized R-hat (Vehtari 2021), not naive R-hat |
| ESS calculation | Custom effective sample size | `az.ess(idata, method='bulk')` | ArviZ implements bulk+tail ESS correctly |
| MCMC sampler | Custom HMC | `numpyro.infer.NUTS` | Already installed, well-tested |
| Prior sampling | Manual loop | `Predictive(model, num_samples=50)(rng_key)` | Correct NumPyro idiom, vectorized |
| MAT file parsing | Custom reader | `scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)` | Handles MATLAB struct nesting |
| Posterior summary | Custom quantile code | `az.summary(idata, var_names=[...])` | Returns DataFrame with mean, sd, hdi, rhat, ess_bulk |

---

## Common Pitfalls (Summary)

### Pitfall: `set_host_device_count` called after JAX import
**What goes wrong:** `numpyro.set_host_device_count(4)` has no effect if JAX is already imported.
**How to avoid:** Set `os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'` before all imports.

### Pitfall: Divergences treated as blocking gate
**What goes wrong:** Model posterior geometry may produce persistent divergences even with 4000 warmup + target_accept=0.99.
**How to avoid:** Per CONTEXT.md decision: document divergence count in JSON, do not gate on zero divergences. Update REQUIREMENTS.md SC-3 wording.

### Pitfall: tau update equation mismatch with paper
**What goes wrong:** The simplified `tau / UU` in `metrics.py` loses the full predictive-variance-weighted update. If 04-02 parameter recovery fails (r < 0.85), this is the first thing to check.
**How to avoid:** Use the full update from `numpyro_models.py` lines 133–138, not the simplified form.

---

## Code Examples

### Canonical NUTS run with diagnostics

```python
# Source: confirmed working with NumPyro 0.19.0, ArviZ 0.23.4, JAX 0.4.35
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='arviz')

import jax
import jax.numpy as jnp
import numpyro
import arviz as az
from numpyro.infer import MCMC, NUTS

kernel = NUTS(model_fn, target_accept_prob=0.95, max_tree_depth=10)
mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000, num_chains=4, progress_bar=False)
mcmc.run(jax.random.PRNGKey(seed), bag_positions=bag_jax, context='changepoint',
         extra_fields=('diverging',))

idata = az.from_numpyro(mcmc)
rhat_max = max(float(az.rhat(idata)[v]) for v in az.rhat(idata).data_vars)
ess_min = min(float(az.ess(idata, method='bulk')[v])
              for v in az.ess(idata, method='bulk').data_vars)
n_div = int(mcmc.get_extra_fields()['diverging'].sum())
```

### Prior sampling for parameter recovery

```python
# Source: NumPyro Predictive API, confirmed with numpyro 0.19.0
from numpyro.infer import Predictive
import jax

prior_sampler = Predictive(reduced_bayesian_model, num_samples=50)
prior_samples = prior_sampler(jax.random.PRNGKey(0))
# prior_samples['H'].shape == (50,), prior_samples['LW'].shape == (50,), etc.
```

### .mat file loading (sliding window fits)

```python
# Source: confirmed working with scipy, direct inspection 2026-04-29
import scipy.io
import numpy as np

mat = scipy.io.loadmat(
    'data/raw/nassar2021/slidingWindowFits_subjects_23-Nov-2021.mat',
    squeeze_me=True,
    struct_as_record=False,
)
data = mat['binRegData']
# data.subRunCoeffsOdd: shape (134, 115, 2) — subjects × windows × [intercept, slope]
# data.subRunCoeffsCP:  shape (134, 115, 2)
odd_coeffs = data.subRunCoeffsOdd   # float64, some NaNs at window edges
cp_coeffs = data.subRunCoeffsCP
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PyEM MAP estimation | NumPyro NUTS full posterior | Phase 1 (completed) | Full uncertainty quantification, posterior predictive checks |
| PyMC with PyTensor backend | NumPyro with JAX | Phase 1 (completed) | Better speed, JAX composability |
| R-hat from `numpyro.diagnostics` | `az.rhat(az.from_numpyro(mcmc))` | Phase 4 (this phase) | Rank-normalized R-hat (Vehtari 2021 standard) |
| ArviZ 0.17.x API | ArviZ 0.23.4 | Phase 4 install | FutureWarning present but all APIs functional |

**Deprecated in this codebase:**
- `archive/bayesian_legacy/pyem_models.py` — keep archived, do not re-import
- `archive/bayesian_pymc/fit_bayesian_pymc.py` — keep archived
- `cluster/batch_fit_bayesian.py` — has broken import, needs TODO updated (not deleted, referenced from archive docs)

---

## Open Questions

1. **Nassar 2021 exact priors**
   - What we know: paper uses "informed prior derived from MLE fits"; regularized MAP estimation
   - What's unclear: exact distributional form (Beta? Gaussian on logit? Uniform?)
   - Recommendation: Download Brain2021Code from sites.brown.edu/mattlab/resources/ before writing 04-01; extract prior from MATLAB `fmincon` bound constraints or supplement. If unavailable: use `Beta(1.5, 8)` for H (matches true hazard rate 0.125 ≈ mean of Beta(1.5, 8) ≈ 0.16), and document the weakly-informative default clearly.

2. **Raw behavioral data acquisition**
   - What we know: Google Drive link confirmed at sites.brown.edu/mattlab/resources/ for Brain2021Code
   - What's unclear: Whether the download includes individual `statusData` MATLAB files or only the summary statistics we already have
   - Recommendation: Plan 04-03 Task 1 = download and verify structure. If only summary stats are in the download, contact Nassar lab directly for raw trial data.

3. **Generative parameters in raw files**
   - What we know: `extract_nassar_trials.py` shows `currentMean` (helicopter position) and `isChangeTrial` are fields in `statusData`
   - What's unclear: Whether `isChangeTrial` is the ground-truth label or an inferred label
   - Impact on 04-02: Does NOT block parameter recovery — we use prior-sampled H/LW/UU and generate synthetic sequences from scratch.

4. **tau update equation exact form**
   - What we know: Two versions exist in the codebase (simplified in metrics.py vs full weighted in numpyro_models.py). The full version matches Loosen 2023 citations.
   - What's unclear: Whether this exactly matches Nassar 2021 supplement's tau equation
   - Recommendation: Flag for 04-02 validation; if recovery fails, compare against Nassar MATLAB code from the download.

5. **ArviZ migration warning scope**
   - What we know: ArviZ 0.23.4 works; FutureWarning on import; all relevant APIs present
   - What's unclear: Whether any specific functions change semantics before Phase 5
   - Recommendation: Pin `arviz>=0.17.0,<0.25.0` in pyproject.toml to prevent unexpected API breaks.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection: `src/nn4psych/bayesian/numpyro_models.py` — JAX-scan model, jax.lax.cond pattern, existing priors
- Direct codebase inspection: `src/nn4psych/utils/metrics.py` lines 196–235 — alpha_t equations (both conditions)
- Direct `.mat` file inspection (Python script run 2026-04-29): `data/raw/nassar2021/slidingWindowFits_subjects_23-Nov-2021.mat` shape (134, 115, 2), `slidingWindowFits_model_23-Nov-2021.mat` shape (134, 115, 3)
- NumPyro 0.19.0 NUTS API (introspected via `inspect.signature(NUTS.__init__)`) — `target_accept_prob` confirmed parameter name
- ArviZ 0.23.4 installed and APIs tested: `az.from_numpyro`, `az.rhat`, `az.ess`, `az.summary` all functional
- JAX 0.4.35: `XLA_FLAGS='--xla_force_host_platform_device_count=4'` confirmed to set `jax.local_device_count() == 4`
- NumPyro `Predictive` prior sampling: confirmed `Predictive(model, num_samples=50)(rng_key)` returns shape (50,) per parameter

### Secondary (MEDIUM confidence)
- Nassar 2021 PMC full text (https://pmc.ncbi.nlm.nih.gov/articles/PMC8041039/) — sigma_N = 20, N=94 patients + 31 controls, task paradigm (100 trials/condition, hazard=0.125), MAP estimation via fmincon
- Phase 3 timing: `output/circuit_analysis/validation_results.json` — 204.5 min GPU for 100-init ensemble (scale estimate for 04-04 compute planning)
- `scripts/data_pipeline/extract_nassar_trials.py` — per-subject raw data structure (statusData, 4 conditions, data cleaning protocol)
- Nassar lab resources page (WebFetch confirmed): sites.brown.edu/mattlab/resources/ — Google Drive link for Brain2021Code data

### Tertiary (LOW confidence — unverified)
- Nassar 2021 exact prior distributions — extracted from PMC text as "informed prior from MLE fits"; full spec requires supplement download; distributions assumed weakly-informative until verified
- H and LW identifiability weakness — inferred from model structure; not empirically validated in this codebase
- Phase 3 GPU training time extrapolation to Phase 4 RNN seeds — estimated from available timing data
- 2-state HMM recommendation for 04-05 — based on JAX-vectorizability analysis and CRP complexity assessment; not from specific published recommendation

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — directly installed and tested in actinf-py-scripts env
- Model equations: HIGH — confirmed from codebase; alpha_t and Omega_t match published equations
- Prior specifications: LOW — paper priors not extracted; supplement required; weakly-informative defaults provided as fallback
- .mat structure: HIGH — directly inspected
- Raw data availability: HIGH (confirmed missing) — critical blocker for 04-03
- Architecture patterns: HIGH — based on working Phase 1 code patterns
- Pitfalls: MEDIUM — some from direct experience (JAX tracing, ArviZ FutureWarning), some inferred

**Research date:** 2026-04-29
**Valid until:** 2026-05-29 (stable domain; ArviZ API in flux but pinnable)
