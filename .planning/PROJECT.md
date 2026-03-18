# nn4psych

## What This Is

An end-to-end computational neuroscience pipeline that trains RNN-based reinforcement learning agents on cognitive tasks, analyzes the resulting network dynamics using latent circuit inference (Langdon & Engel 2025) and fixed point analysis, and fits Bayesian normative models (Nassar 2021) to both human schizophrenia data and RNN agent outputs. The project bridges neural network models of cognition with Bayesian observer models to study how predictive inference mechanisms differ across populations and model architectures.

## Core Value

The RNN-RL agent must train on multiple cognitive tasks and produce analyzable hidden representations that can be compared against human behavioral data through both circuit-level (latent circuit inference) and computational-level (Bayesian model fitting) analyses.

## Requirements

### Validated

- ✓ ActorCritic RNN model (single-task, PyTorch) — existing
- ✓ MultiTaskActorCritic with task-specific heads — existing
- ✓ PIE_CP_OB_v2 environment (change-point/oddball helicopter task) — existing
- ✓ Configuration system (dataclass-based ExperimentConfig with YAML/JSON) — existing
- ✓ Behavior extraction pipeline (extract_behavior, state collection) — existing
- ✓ Learning rate metrics (get_lrs_v2, prediction error) — existing
- ✓ Model save/load utilities — existing
- ✓ Numbered data pipeline scripts (01-07) — existing
- ✓ NeuroGym task integration (DawTwoStep, SingleContextDecisionMaking) — existing
- ✓ Nassar 2021 human schizophrenia data (.mat files) — existing
- ✓ src/ layout packaging (PEP 517) — existing

### Active

- [ ] Verify RNN-RL training works end-to-end on PIE environment
- [ ] Verify RNN-RL training on NeuroGym tasks (DawTwoStep, SingleContextDecisionMaking)
- [ ] Add context-dependent decision-making task (Mante et al. color/motion discrimination)
- [ ] Verify RNN training on context-dependent DM task
- [ ] Implement fixed point analysis for trained RNNs
- [ ] Implement latent circuit inference (Langdon & Engel 2025: Q embedding, w_rec, w_in, w_out fitting)
- [ ] Implement latent circuit connectivity perturbation analysis
- [ ] Archive PyEM Bayesian models to archive/
- [ ] Set up Nassar 2021 reduced Bayesian model on NumPyro/JAX
- [ ] Fit Nassar 2021 model to original schizophrenia data (.mat files)
- [ ] Fit Nassar 2021 model to RNN agent behavioral outputs
- [ ] Compare Bayesian model fits across human vs RNN agent data
- [ ] Reorganize project structure to match project_utils conventions
- [ ] Clean up scripts and documentation

### Out of Scope

- Full PyMC Bayesian implementation — archiving in favor of NumPyro/JAX
- PyEM Bayesian models — archiving, replaced by NumPyro
- GPU-optimized distributed training — CPU sufficient for current model sizes
- Real-time or interactive visualization — static figures sufficient
- Mobile or web deployment — research tool only

## Context

- The project builds on Nassar et al. 2021 which studies predictive inference in schizophrenia using a helicopter/bag positioning paradigm with change-point and oddball conditions
- The Langdon & Engel 2025 paper (Nature Neuroscience) introduces latent circuit inference: fitting a low-dimensional latent circuit model (y = Qx, with ReLU dynamics) to RNN hidden states to extract interpretable circuit mechanisms
- The context-dependent decision-making task (Mante et al. 2013) involves discriminating color or motion based on a context cue — used as the primary task in Langdon & Engel 2025
- Existing Bayesian code spans three implementations: PyMC (scripts/analysis/bayesian/), PyEM (bayesian/pyem_models.py), and NumPyro (bayesian/numpyro_models.py) — consolidating to NumPyro/JAX only
- The project_utils repo at C:\Users\aman0087\Documents\Github\project_utils defines standard project conventions for data analysis: numbered pipeline scripts, data/raw → data/processed flow, src/ layout
- Known tech debt: hardcoded dimensions in analysis functions, untested multi-task training loop, fragile extract_behavior relying on private env methods

## Constraints

- **Tech stack (NN)**: PyTorch for RNN models — existing codebase, not switching
- **Tech stack (Bayesian)**: NumPyro/JAX for Bayesian models — replacing PyMC and PyEM
- **Data**: Nassar 2021 schizophrenia data already in data/raw/nassar2021/ (.mat format)
- **Structure**: Must follow project_utils conventions (src/ layout, numbered pipelines, data organization)
- **Reference papers**: Langdon & Engel 2025 (latent circuit inference), Nassar et al. 2021 (Bayesian observer model, schizophrenia data)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| NumPyro/JAX over PyMC for Bayesian models | JAX backend is faster, composable with NumPy, better for HMC sampling | — Pending |
| Archive PyEM models rather than maintain | PyEM is legacy, NumPyro replaces it cleanly | — Pending |
| Latent circuit inference over standard dim reduction | Langdon & Engel 2025 provides interpretable circuit mechanisms, not just correlations | — Pending |
| Add context-dependent DM task | Required for latent circuit inference analysis (primary task in Langdon & Engel 2025) | — Pending |

---
*Last updated: 2026-03-18 after initialization*
