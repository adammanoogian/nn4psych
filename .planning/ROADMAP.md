# Roadmap: nn4psych

## Overview

nn4psych trains RNN-RL agents on cognitive tasks, extracts circuit-level mechanisms via latent circuit inference, and compares agent behavior to human schizophrenia data through Bayesian model fitting. The pipeline runs five sequential phases: clean up the project structure and fix blocking bugs, verify all RNN training pipelines, implement latent circuit inference with full validation, run Bayesian model fitting on both human and RNN data, and produce the final human-versus-RNN behavioral comparison.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Infrastructure and Organization** - Align project structure, fix blocking bugs, update dependencies
- [x] **Phase 2: RNN Training Verification** - Verify end-to-end training on all cognitive tasks including context-DM
- [ ] **Phase 3: Latent Circuit Inference** - Implement and validate latent circuit fitting with 100-init ensemble
- [ ] **Phase 4: Bayesian Model Fitting** - Fit Nassar 2021 model to human and RNN behavioral data with full diagnostics
- [ ] **Phase 5: Human-vs-RNN Comparison** - Extract learning rates, run group-level comparison, produce final outputs

## Phase Details

### Phase 1: Infrastructure and Organization
**Goal**: The project structure is aligned with project_utils conventions, all blocking bugs are fixed, and the dependency stack is correct so that all subsequent phases can run without structural obstacles.
**Depends on**: Nothing (first phase)
**Requirements**: ORG-01, ORG-02, ORG-03, ORG-04
**Success Criteria** (what must be TRUE):
  1. `import nn4psych.bayesian` succeeds from a clean install (bayesian/ consolidated into src/nn4psych/bayesian/)
  2. `python -c "import jax; print(jax.default_backend())"` prints `cpu` without GPU allocation error when PyTorch is also imported in the same process
  3. A test with oddball condition runs through the NumPyro scan without silently using changepoint equations (JAX tracing bug resolved)
  4. `extract_behavior` runs on a NeuroGym environment using only public gym API without AttributeError
  5. `pip install -e ".[bayesian]"` installs JAX 0.9.1, NumPyro 0.20.0, ArviZ 1.0.0 and no PyMC/PyTensor
**Plans:** 3 plans

Plans:
- [ ] 01-01-PLAN.md — Move bayesian/ to src/nn4psych/bayesian/, create NumPyro-only __init__.py with JAX CPU enforcement, update script imports (Wave 1)
- [ ] 01-02-PLAN.md — Update pyproject.toml deps (NumPyro/JAX, remove PyMC), bump Python >=3.11, archive PyMC/PyEM files (Wave 1)
- [ ] 01-03-PLAN.md — Fix JAX tracing bug (jax.lax.cond), add reset_epoch() to envs, fix extract_behavior private API, parameterize batch dims (Wave 2)

### Phase 2: RNN Training Verification
**Goal**: The RNN ActorCritic trains and converges on all three task types (PIE, NeuroGym tasks, context-DM), and behavior and hidden states can be extracted for downstream analysis.
**Depends on**: Phase 1
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04
**Success Criteria** (what must be TRUE):
  1. Training on PIE_CP_OB_v2 runs to completion and reward curves show learning (not flat or diverging)
  2. Training on DawTwoStep and SingleContextDecisionMaking runs to completion without error
  3. ContextDecisionMaking-v0 task loads from NeuroGym with a gym-compatible interface and the agent receives correct observations
  4. Training on context-DM task runs to completion and hidden state arrays save to data/processed/rnn_behav/ as .npy files
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — Fix 4 blocking bugs (main guard, NeuroGym state crash, GAE tensor, obs_dim mismatch), add neurogym dependency (Wave 1)
- [ ] 02-02-PLAN.md — Implement extract_behavior_with_hidden(), verify PIE + NeuroGym training end-to-end (Wave 2)
- [ ] 02-03-PLAN.md — Create context-DM training script, train on ContextDecisionMaking-v0, save hidden states to data/processed/rnn_behav/ (Wave 2)

### Phase 3: Latent Circuit Inference
**Goal**: The latent circuit inference pipeline fits Q, w_rec, w_in, w_out from context-DM RNN hidden states with 100-initialization ensemble validation, and the inferred circuit passes both activity-level and connectivity-level checks.
**Depends on**: Phase 2
**Requirements**: CIRC-01, CIRC-02, CIRC-03, CIRC-04, CIRC-05
**Success Criteria** (what must be TRUE):
  1. The fitting routine runs 100+ random initializations and selects the best solution by reconstruction loss
  2. The selected solution's QᵀW_recQ correlation with inferred w_rec is >= 0.85 (invariant subspace condition)
  3. Projecting RNN responses onto latent axes reproduces the trial-averaged dynamics (activity-level validation passes)
  4. Perturbing w_rec in latent space and mapping back to RNN weights produces a measurable predicted behavioral change
**Plans**: TBD

Plans:
- [ ] 03-01: Implement src/nn4psych/analysis/circuit_inference.py with Q/w_rec/w_in/w_out optimization and 100-init ensemble loop
- [ ] 03-02: Implement invariant subspace validation (QᵀW_recQ correlation check) and activity-level/connectivity-level validation
- [ ] 03-03: Implement perturbation analysis (latent connectivity perturbation → RNN weight perturbation → behavioral prediction); wire scripts/data_pipeline/08_infer_latent_circuits.py

### Phase 4: Bayesian Model Fitting
**Goal**: The Nassar 2021 reduced Bayesian observer model is implemented in NumPyro/JAX, validated on synthetic data, fit per-subject to human schizophrenia data, and fit to RNN behavioral outputs — all with MCMC convergence diagnostics enforced.
**Depends on**: Phase 1 (JAX infrastructure), Phase 2 (RNN behavioral outputs)
**Requirements**: BAYES-01, BAYES-02, BAYES-03, BAYES-04, BAYES-05, BAYES-06
**Success Criteria** (what must be TRUE):
  1. PyEM models are archived to archive/ and no longer importable from the main package
  2. Parameter recovery simulation on 50 synthetic datasets shows recovered vs true parameter correlation >= 0.85 for all parameters
  3. Per-subject MCMC fits on human schizophrenia data complete with R-hat <= 1.01, ESS >= 400, and zero divergences
  4. Per-model MCMC fits on RNN behavioral outputs complete with the same convergence criteria
**Plans**: TBD

Plans:
- [ ] 04-01: Archive PyEM models; implement src/nn4psych/bayesian/ subpackage with Nassar 2021 reduced model in NumPyro/JAX
- [ ] 04-02: Implement MCMC convergence diagnostics (R-hat, ESS, trace plots via ArviZ) and parameter recovery simulation (50 synthetic datasets)
- [ ] 04-03: Fit model to Nassar 2021 human .mat files (per-subject, per-condition); wire scripts/data_pipeline/09_fit_human_data.py
- [ ] 04-04: Fit model to RNN behavioral outputs; wire scripts/data_pipeline/10_fit_rnn_data.py

### Phase 5: Human-vs-RNN Comparison
**Goal**: Learning rates are extracted from human data and RNN outputs on a comparable basis, and a group-level statistical comparison (schizophrenia vs control vs RNN) is produced with WAIC model comparison.
**Depends on**: Phase 4
**Requirements**: COMP-01, COMP-02, COMP-03
**Success Criteria** (what must be TRUE):
  1. Learning rate time series are extracted from Nassar 2021 .mat files using the same three-bin scheme (non-update, moderate, total) applied to RNN outputs
  2. A group-level comparison table and figure show Bayesian parameter distributions for schizophrenia, control, and RNN populations
  3. WAIC model comparison values are computed and reported alongside the group-level parameter comparison
**Plans**: TBD

Plans:
- [ ] 05-01: Implement src/nn4psych/analysis/comparison.py with learning rate extraction from .mat files and learning rate binning
- [ ] 05-02: Implement group-level statistical comparison and WAIC; produce comparison figure and summary table

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Infrastructure and Organization | 3/3 | ✓ Complete | 2026-03-18 |
| 2. RNN Training Verification | 3/3 | ✓ Complete | 2026-03-19 |
| 3. Latent Circuit Inference | 0/3 | Not started | - |
| 4. Bayesian Model Fitting | 0/4 | Not started | - |
| 5. Human-vs-RNN Comparison | 0/2 | Not started | - |
