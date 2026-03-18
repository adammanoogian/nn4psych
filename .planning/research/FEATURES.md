# Feature Landscape: Computational Neuroscience RNN Analysis Pipeline

**Domain:** Computational neuroscience — RNN-RL training, latent circuit inference, Bayesian cognitive model fitting, human-model behavioral comparison
**Researched:** 2026-03-18
**Milestone:** Subsequent — latent circuit inference, NumPyro Bayesian fitting, new task environments
**Overall confidence:** MEDIUM-HIGH (core features verified against Langdon & Engel 2025, Nassar 2021, FixedPointFinder toolbox, NumPyro/ArviZ docs)

---

## Table Stakes

Features users expect in a credible analysis pipeline. Missing = pipeline is not publishable or reproducible.

### 1. Latent Circuit Inference (Low-Rank RNN Analysis)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Q embedding matrix fitting | Core of Langdon & Engel 2025 — projects high-dim RNN hidden states to low-dim latent space | High | Nonlinear least squares over orthonormal subspaces; requires custom optimizer |
| Recurrent connectivity fitting (w_rec) | Required to recover latent circuit dynamics from RNN weights | High | Conjugation of RNN weight matrix with embedding; validates inferred mechanism |
| Input/output connectivity fitting (w_in, w_out) | Required for full latent circuit model (inputs → latent → behavior) | Medium | Coupled to w_rec; joint optimization |
| Reconstruction error as loss | Standard way to validate latent circuit fit quality | Low | Measures how well LatentNet reproduces RNN trajectories |
| Activity-level validation | Projecting RNN responses onto latent axes to verify mechanistic signatures | Medium | E.g., suppression pattern in sensory representations for context-DM task |
| Connectivity-level validation | Confirming predicted low-rank structure exists in full RNN weight matrix | Medium | "Conjugate the RNN connectivity matrices with the embedding matrix" |
| Perturbation analysis | Translating latent connectivity changes to high-dim weight perturbations, predicting behavioral effects | High | Causal evidence for inferred mechanism; differentiating feature but expected by field |
| Trial-averaged hidden state extraction | Raw material for latent circuit fitting | Low | Condition-averaged activity across task epochs (stimulus, delay, response) |

**Dependency:** Q, w_rec, w_in, w_out fitting all depend on trial-averaged hidden state extraction from a trained RNN. Perturbation analysis depends on connectivity fitting. Validation depends on fitting.

**Reference:** Langdon & Engel 2025 (Nature Neuroscience), engellab/latentcircuit GitHub. MEDIUM-HIGH confidence.

---

### 2. NumPyro Bayesian Cognitive Model Fitting

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Prior specification (H, LW, UU, sigma_motor, sigma_LR) | Model must be specified before fitting; priors are part of generative model | Low | Already partially implemented in bayesian/numpyro_models.py |
| NUTS/HMC sampling | Standard for full posterior inference in continuous-parameter models; NumPyro's default | Low | Already in codebase via NUTS(normative_model) + MCMC |
| Multi-chain sampling (n_chains >= 4) | Required to detect non-convergence via R-hat | Low | Already in run_mcmc() |
| R-hat convergence diagnostic | Field-standard: R-hat < 1.01 required for credible posterior | Low | ArviZ az.rhat(); not yet implemented in pipeline |
| Effective Sample Size (ESS) | Required alongside R-hat; ESS > 100/chain minimum | Low | ArviZ az.ess(); not yet implemented |
| Trace plots | Visual convergence check; expected in any MCMC workflow | Low | ArviZ az.plot_trace(); not yet implemented |
| Posterior summary (mean, SD, HPDI) | Reporting standard for Bayesian parameter estimates | Low | Already in summarize_posterior() |
| Posterior predictive check (PPC) | Core model validation: do simulated data resemble observed data? | Medium | Already in posterior_predictive(); needs plotting layer |
| Separate fits per subject | Bayesian model fitting is done per-subject (individual differences) | Medium | Needs loop over subjects + result aggregation |
| Separate fits per condition (changepoint vs oddball) | Nassar 2021 fits both conditions; model comparison tests context-specificity | Medium | Context flag already in normative_model() |
| Fit to human data (.mat files) | Primary scientific goal — fit Nassar model to Nassar 2021 schizophrenia dataset | Medium | Requires .mat loading + reshaping to bucket/bag arrays |
| Fit to RNN agent behavioral outputs | Secondary goal — compare RNN-derived parameters to human parameters | Medium | Requires extract_behavior output → bucket/bag format |

**Dependency:** Convergence diagnostics (R-hat, ESS, trace plots) depend on completed MCMC runs. Per-subject fits depend on data loading. PPC depends on posterior samples.

**Reference:** NumPyro docs, ArviZ docs, Nassar 2021 (PMC8041039). HIGH confidence for MCMC/ArviZ standards; MEDIUM for Nassar-specific parameter structure.

---

### 3. Fixed Point Analysis

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Fixed point finder (gradient descent from random inits) | Standard first step in dynamical systems analysis of RNNs | Medium | Existing FixedPointAnalyzer in analyze_fixed_points.py; uses simple iteration, not Golub/Sussillo optimizer |
| Jacobian computation at fixed points | Required for stability analysis; linearizes dynamics locally | Low | Already in compute_jacobian() |
| Stability classification (stable / unstable / saddle) | Standard output: eigenvalue magnitude > 1 = unstable | Low | Already in analyze_stability(); should add saddle detection |
| PCA visualization of fixed points + trajectories | Field-standard visualization: fixed points overlaid on PC-reduced trajectory space | Low | Partially in codebase; FixedPointFinder generates these natively |
| Input-conditioned fixed points | Fixed points computed for each task input condition, not just zero input | Medium | Critical for context-DM task: different contexts → different fixed points |
| Line attractor detection (null space analysis) | Important for working memory / integration tasks | Medium | Already in check_line_attractor() |
| Trajectory simulation from arbitrary initial states | Required to visualize attractor basins | Low | Already in simulate_trajectory() |
| Unstable modes visualization | Lines from saddle points showing transition directions | Medium | Not yet in codebase; FixedPointFinder provides this |

**Dependency:** All fixed point analysis depends on a trained RNN. Input-conditioned analysis depends on having task input vectors. Visualization depends on fixed point finding.

**Reference:** Golub & Sussillo 2018 (FixedPointFinder, JOSS), existing analyze_fixed_points.py. HIGH confidence.

---

### 4. Behavioral Comparison: Models vs Human Data

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Trial-level learning rate extraction (RNN) | Primary behavioral metric in Nassar paradigm | Low | Already in get_lrs_v2(); returns per-trial LR |
| Trial-level learning rate extraction (human) | Required for direct comparison | Low | Requires parsing Nassar .mat files |
| Learning rate binning / categorization | Nassar 2021 reports non-updates (<0.1), moderate (0.1-0.9), total (≥0.9) | Low | Needs implementation |
| Group-level parameter comparison (patient vs control) | Core scientific question: do schizophrenia parameters differ from controls? | Medium | Requires per-subject fits + statistical comparison |
| Statistical tests for group differences | Field expectation: t-test, Mann-Whitney, or Bayesian comparison | Low | Not yet in pipeline |
| Model information criteria (BIC, AIC, WAIC) | Required to compare model variants (e.g., changepoint vs oddball fits, or full vs reduced model) | Low | BIC/AIC in model_comparison.py; WAIC in numpyro_models.py |
| Behavioral metric visualization (learning rate histograms, update distributions) | Required for paper figures | Low | Partially in visualize_learning_rates.py |
| Parameter correlation with clinical scores | Nassar 2021: correlates H, LW, UU with BPRS, SANS, MATRICS | Medium | Requires clinical metadata from .mat files; important for validation |

**Dependency:** Human-RNN comparison depends on both human data loading and RNN behavior extraction. Group comparison depends on per-subject fits. Clinical correlation depends on metadata extraction.

**Reference:** Nassar 2021 (PMC8041039 analysis pipeline). HIGH confidence.

---

### 5. Task Environment (Context-Dependent Decision Making)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| ContextDecisionMaking-v0 task (Mante 2013 paradigm) | Primary task in Langdon & Engel 2025; required for latent circuit inference | Medium | Available in neurogym as ContextDecisionMaking-v0 |
| Two-modality stimulus input (color + motion) | Core task structure | Low | Part of ContextDecisionMaking environment |
| Context cue input (rule signal) | Tells agent which modality to use | Low | Part of ContextDecisionMaking environment |
| Gym-compatible interface (step/reset/observation_space) | Required to integrate with existing ActorCritic training loop | Low | neurogym environments follow gym API |
| Reward-based training compatibility | Actor-critic training requires reward signal | Low | Task must return reward; neurogym supports RL mode |

**Dependency:** All latent circuit analysis depends on RNN trained on context-DM task. Context-DM task integration depends on neurogym compatibility with training loop.

**Reference:** neurogym ContextDecisionMaking-v0 docs, Mante et al. 2013 (Nature), Langdon & Engel 2025. HIGH confidence.

---

## Differentiators

Features that set this pipeline apart. Not expected in every project, but scientifically valuable and novel.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Cross-population latent circuit comparison | Compare latent circuit parameters (Q, w_rec) across model variants trained with different hyperparameters — tests whether circuit mechanisms are robust | High | Novel; not in standard latentcircuit toolbox |
| Bayesian parameter recovery validation | Fit NumPyro model to synthetic data with known ground truth parameters to verify identifiability | Medium | Standard in cognitive modeling but often skipped; Nassar 2021 does this |
| RNN-human parameter overlay plots | Directly visualize RNN agent's Bayesian parameters against human schizophrenia / control distributions | Medium | Core scientific contribution; bridges circuit and computational levels |
| Context error parameters (Nassar 2021 extended model) | Proportion context error parameters capture schizophrenia-specific behavior — mixing changepoint/oddball rules | Medium | Nassar 2021 10-parameter model; more informative than base 5-parameter model |
| Latent circuit perturbation → behavioral prediction | Perturb inferred latent connectivity, re-simulate RNN behavior, test behavioral change — causal evidence | High | Langdon & Engel 2025 core method; strong mechanistic claim |
| Multi-task latent circuit comparison | Fit latent circuits to RNN trained on PIE task AND context-DM task, compare circuit mechanisms | High | Novel; tests whether same mechanisms generalize |
| dPCA / demixed PCA of RNN hidden states | Goes beyond plain PCA: isolates stimulus, context, and choice-related subspaces | High | More interpretable than PCA for cognitive tasks; common in papers using Mante task |
| Leave-one-subject-out cross-validation for Bayesian fits | Validates that individual parameter fits generalize; required for clinical classification claims | Medium | Nassar 2021 does this; important for reproducibility |

**Complexity note:** High = multiple weeks of implementation + validation. Medium = 1-2 weeks. Low = days.

---

## Anti-Features

Features to deliberately NOT build. Common mistakes in this domain.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| PyEM / PyMC Bayesian implementations | Already in codebase; adds maintenance burden; NumPyro is faster and cleaner | Archive PyEM/PyMC to archive/; use NumPyro only |
| GPU-distributed RNN training | Current model sizes (hidden_dim=64, N<200 trials) do not require GPU; overhead > benefit | CPU training is sufficient; add GPU only if scaling to hidden_dim>512 |
| Interactive / real-time visualization | Research tool; no deployment requirement; static figures are sufficient and reproducible | Use matplotlib/seaborn for static figures; export to PDF |
| Full-population neural recording interface | This pipeline operates on RNN-simulated "neural" data, not real recording data | Keep inputs as RNN hidden states; do not generalize to raw electrophysiology |
| Continuous integration for scientific correctness | Numerical outputs change with random seed; CI will be brittle | Use validation scripts with fixed seeds and expected-range checks; not strict equality |
| Automatic hyperparameter tuning for Bayesian priors | Prior sensitivity analysis is scientifically important but automated tuning is fishing | Run explicit prior sensitivity checks manually; document prior choices |
| Custom MCMC sampler | NumPyro NUTS is state-of-the-art; building a custom sampler wastes time | Use NUTS with JAX JIT; only switch to SVI if NUTS is too slow |
| Web dashboard for results | Adds engineering complexity with no scientific value | Use Jupyter notebooks + static figures |
| Stochastic latent circuit fitting (variational) | Valente et al. 2022 / 2024 exist but require different architecture; adds complexity beyond scope | Stick to Langdon & Engel 2025 deterministic embedding approach |

---

## Feature Dependencies

```
Task Environment (context-DM)
    → RNN Training (context-DM)
        → Trial-averaged hidden states
            → Latent Circuit Fitting (Q, w_rec, w_in, w_out)
                → Activity-level validation
                → Connectivity-level validation
                → Perturbation analysis → Behavioral prediction

RNN Training (any task)
    → Behavior extraction (extract_behavior)
        → Learning rate extraction (get_lrs_v2)
            → Learning rate comparison (RNN vs human)
        → Bucket/bag position arrays
            → NumPyro MCMC fit
                → R-hat / ESS / trace plots (convergence check)
                → Posterior summary (H, LW, UU, sigma_motor, sigma_LR)
                → Posterior predictive check
                → WAIC / model comparison

Human data (.mat files)
    → Learning rate extraction (human)
        → Learning rate comparison (RNN vs human)
    → NumPyro MCMC fit (per subject)
        → Group comparison (schizophrenia vs control)
        → Clinical correlation (BPRS, SANS, MATRICS)

Fixed Point Analysis (depends on trained RNN):
    → Fixed point finding
        → Jacobian + stability classification
            → PCA visualization (fixed points + trajectories)
            → Unstable modes visualization
    → Line attractor detection
    → Input-conditioned fixed points (depends on task input vectors)
```

---

## MVP Recommendation

For the current milestone (adding latent circuit inference, NumPyro Bayesian fitting, context-DM task), prioritize:

1. Context-DM task integration (ContextDecisionMaking-v0 via neurogym) — everything downstream depends on this
2. Latent circuit fitting core (Q, w_rec, w_in, w_out optimization loop) — primary scientific deliverable
3. Activity-level + connectivity-level validation — minimum required to claim the inference worked
4. NumPyro fit to human data (.mat) with convergence diagnostics (R-hat, ESS, trace plots) — required for any publishable Bayesian analysis
5. NumPyro fit to RNN behavioral outputs — enables the human-vs-RNN comparison
6. Learning rate comparison (group-level: schizophrenia vs control vs RNN) — core scientific question

Defer to post-MVP:
- **Perturbation analysis**: High complexity; requires validated latent circuit first. Do after connectivity validation.
- **dPCA**: High complexity; nice-to-have for latent circuit visualization but not required for core comparisons.
- **Leave-one-subject-out CV**: Medium complexity; important for clinical classification but defer until basic fits are validated.
- **Multi-task latent circuit comparison**: High complexity; novel contribution but requires stable single-task pipeline first.
- **Clinical correlation (BPRS, SANS, MATRICS)**: Medium complexity; requires clinical metadata parsing; defer until parameter estimates are stable.

---

## Sources

- Langdon & Engel 2025 (Nature Neuroscience): https://www.nature.com/articles/s41593-025-01869-7
- engellab/latentcircuit GitHub: https://github.com/engellab/latentcircuit
- Langdon & Engel 2025 PMC full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11893458/ [HIGH confidence]
- Golub & Sussillo 2018 FixedPointFinder (JOSS): https://github.com/mattgolub/fixed-point-finder [HIGH confidence]
- Nassar 2021 schizophrenia analysis (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC8041039/ [HIGH confidence, fetched full text]
- NumPyro documentation: http://pyro.ai/numpyro/ [HIGH confidence]
- ArviZ MCMC diagnostics: https://arviz-devs.github.io/EABM/Chapters/MCMC_diagnostics.html [HIGH confidence]
- neurogym ContextDecisionMaking-v0: https://neurogym.github.io/envs/ContextDecisionMaking-v0.html [MEDIUM confidence — docs incomplete]
- Valente et al. 2022 NeurIPS (low-rank RNN extraction): https://proceedings.neurips.cc/paper_files/paper/2022/hash/9877d915a4b4f00e85e7b4cfdf41e450-Abstract-Conference.html [MEDIUM confidence — fetched PDF failed]
- Existing codebase: bayesian/numpyro_models.py, scripts/analysis/analyze_fixed_points.py [HIGH confidence — direct inspection]
