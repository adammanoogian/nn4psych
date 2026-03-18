# Requirements: nn4psych

**Defined:** 2026-03-18
**Core Value:** RNN agent trainable on multiple cognitive tasks with analyzable hidden representations comparable to human data via Bayesian model fitting

## v1 Requirements

### RNN Training & Verification

- [ ] **TRAIN-01**: RNN ActorCritic trains end-to-end on PIE_CP_OB_v2 environment (change-point and oddball conditions)
- [ ] **TRAIN-02**: RNN ActorCritic trains on NeuroGym tasks (DawTwoStep, SingleContextDecisionMaking)
- [ ] **TRAIN-03**: ContextDecisionMaking-v0 task integrated from neurogym with gym-compatible interface
- [ ] **TRAIN-04**: RNN ActorCritic trains on context-dependent decision-making task

### Latent Circuit Inference

- [ ] **CIRC-01**: Latent circuit model fits Q embedding, w_rec, w_in, w_out from RNN hidden states
- [ ] **CIRC-02**: Fitting runs 100+ random initializations with convergence selection
- [ ] **CIRC-03**: Activity-level validation (project RNN responses onto latent axes)
- [ ] **CIRC-04**: Connectivity-level validation (Q^T W_rec Q matches inferred w_rec)
- [ ] **CIRC-05**: Perturbation analysis translates latent connectivity changes to RNN weight perturbations with behavioral predictions

### Bayesian Model Fitting

- [ ] **BAYES-01**: PyEM models archived to archive/
- [ ] **BAYES-02**: Nassar 2021 reduced Bayesian observer model implemented in NumPyro/JAX
- [ ] **BAYES-03**: MCMC convergence diagnostics (R-hat <= 1.01, ESS >= 400, trace plots)
- [ ] **BAYES-04**: Model fit to Nassar 2021 human schizophrenia data (.mat files, per-subject)
- [ ] **BAYES-05**: Model fit to RNN agent behavioral outputs (per-model)
- [ ] **BAYES-06**: Parameter recovery simulation validates model identifiability with synthetic data

### Human-vs-RNN Comparison

- [ ] **COMP-01**: Learning rate extraction from human data (.mat files)
- [ ] **COMP-02**: Learning rate binning (non-update, moderate, total) for both human and RNN
- [ ] **COMP-03**: Group-level statistical comparison (schizophrenia vs control vs RNN parameters)

### Project Organization

- [ ] **ORG-01**: Directory structure aligned with project_utils conventions
- [ ] **ORG-02**: bayesian/ consolidated into src/nn4psych/bayesian/ subpackage
- [ ] **ORG-03**: pyproject.toml updated (JAX/NumPyro deps, Python >= 3.11, extras)
- [ ] **ORG-04**: Known bugs fixed (JAX tracing in numpyro_models, extract_behavior private API, hardcoded dimensions)

## v2 Requirements

### Dynamical Systems Analysis

- **FP-01**: Fixed point analysis promoted from script to src/nn4psych/analysis/fixed_points.py
- **FP-02**: Input-conditioned fixed points computed for each task condition
- **FP-03**: Unstable modes visualization from saddle points

### Advanced Analysis

- **ADV-01**: dPCA / demixed PCA isolating stimulus, context, choice subspaces
- **ADV-02**: Multi-task latent circuit comparison across PIE and context-DM tasks
- **ADV-03**: Clinical correlation of Bayesian parameters with BPRS, SANS, MATRICS scores
- **ADV-04**: Leave-one-subject-out cross-validation for Bayesian fits
- **ADV-05**: Multi-task training verification (MultiTaskActorCritic on interleaved tasks)

## Out of Scope

| Feature | Reason |
|---------|--------|
| PyEM / PyMC Bayesian implementations | Replaced by NumPyro/JAX; archiving existing code |
| GPU-distributed RNN training | Model sizes (hidden_dim=64) don't require GPU |
| Interactive / real-time visualization | Static matplotlib/seaborn figures sufficient for research |
| Custom MCMC sampler | NumPyro NUTS is state-of-the-art |
| Stochastic latent circuit fitting (variational) | Langdon & Engel 2025 deterministic approach is sufficient |
| Web dashboard | No deployment requirement; Jupyter + static figures |
| Full-population neural recording interface | Pipeline operates on RNN hidden states, not real electrophysiology |
| Automatic hyperparameter tuning for Bayesian priors | Prior sensitivity analysis should be manual and documented |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ORG-01 | Phase 1 | Pending |
| ORG-02 | Phase 1 | Pending |
| ORG-03 | Phase 1 | Pending |
| ORG-04 | Phase 1 | Pending |
| TRAIN-01 | Phase 2 | Pending |
| TRAIN-02 | Phase 2 | Pending |
| TRAIN-03 | Phase 2 | Pending |
| TRAIN-04 | Phase 2 | Pending |
| CIRC-01 | Phase 3 | Pending |
| CIRC-02 | Phase 3 | Pending |
| CIRC-03 | Phase 3 | Pending |
| CIRC-04 | Phase 3 | Pending |
| CIRC-05 | Phase 3 | Pending |
| BAYES-01 | Phase 4 | Pending |
| BAYES-02 | Phase 4 | Pending |
| BAYES-03 | Phase 4 | Pending |
| BAYES-04 | Phase 4 | Pending |
| BAYES-05 | Phase 4 | Pending |
| BAYES-06 | Phase 4 | Pending |
| COMP-01 | Phase 5 | Pending |
| COMP-02 | Phase 5 | Pending |
| COMP-03 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0

---
*Requirements defined: 2026-03-18*
*Last updated: 2026-03-18 — traceability complete after roadmap creation*
