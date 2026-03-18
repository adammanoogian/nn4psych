# Project Research Summary

**Project:** nn4psych — Computational Neuroscience RNN Analysis Pipeline
**Domain:** Computational neuroscience — RNN-RL training, latent circuit inference, Bayesian cognitive model fitting, human-model behavioral comparison
**Researched:** 2026-03-18
**Confidence:** HIGH (stack and architecture), MEDIUM-HIGH (features and pitfalls)

## Executive Summary

nn4psych is a research-grade computational neuroscience pipeline that trains reinforcement-learning RNNs on cognitive tasks, extracts their computational mechanisms via latent circuit inference, and compares their behavior against human participants using Bayesian cognitive models. The project already has a functioning PyTorch RNN-RL training loop, NeuroGym task integration, and a partially-implemented NumPyro Bayesian module. The next milestone adds four interconnected capabilities: a context-dependent decision-making task (ContextDecisionMaking-v0), latent circuit inference following Langdon & Engel 2025, fixed point dynamical analysis, and full-pipeline Bayesian model fitting with MCMC diagnostics and human data comparison.

The recommended approach is a strictly sequential five-stage pipeline: train RNN on context-DM task, extract behavioral and hidden-state outputs, run latent circuit inference and fixed point analysis (both pure NumPy/SciPy, no new framework), then run NumPyro/JAX Bayesian fitting, and finally compare RNN parameters to human Nassar 2021 data. All framework handoffs cross the PyTorch-to-JAX boundary via NumPy arrays saved to `data/processed/` — this pattern is already in use in the existing codebase and must be maintained consistently. JAX must be forced to CPU via `JAX_PLATFORM_NAME=cpu` to prevent GPU memory conflicts with PyTorch.

The two critical risks are scientific, not technical: latent circuit inference requires running at least 100 random initializations and validating the invariant subspace condition (correlation between QᵀW_recQ and inferred w_rec >= 0.85) before treating any solution as meaningful, and the NumPyro model has a confirmed bug where a Python string `context` argument inside `jax.lax.scan` silently ignores the oddball condition. Both failures produce results that look plausible while being scientifically incorrect. Fix the JAX tracing bug and build the multi-init loop with validation before any scientific interpretation.

---

## Key Findings

### Recommended Stack

The project requires no new major frameworks beyond what is already present or partially declared. PyTorch 2.10.0 handles all RNN training and latent circuit optimization (the engellab/latentcircuit reference implementation is pure PyTorch). NumPyro 0.20.0 + JAX 0.9.1 + ArviZ 1.0.0 handle Bayesian fitting — the core model is already implemented in `bayesian/numpyro_models.py`. NeuroGym 2.3.0 provides `ContextDecisionMaking-v0` directly. Fixed point analysis requires only SciPy's L-BFGS-B optimizer and `torch.autograd.functional.jacobian` — no third-party fixed-point library is safe to use (the only maintained PyTorch option, mattgolub/fixed-point-finder, has broken tests since 2022).

The critical pyproject.toml change is replacing the `bayesian` optional dependency group: remove PyMC and PyTensor, add JAX 0.9.1, NumPyro 0.20.0, ArviZ 1.0.0. Python minimum must be bumped to 3.11 (required by JAX 0.9.1). Also add Pandas 2.0.x and Seaborn 0.13.2 to support latent circuit output reporting.

**Core technologies:**
- PyTorch 2.10.0: RNN training and latent circuit optimization — official engellab implementation is pure PyTorch, no porting needed
- JAX 0.9.1 (CPU only) + NumPyro 0.20.0: Bayesian MCMC fitting — already partially implemented, must be CPU-forced to avoid GPU conflict
- ArviZ 1.0.0: MCMC diagnostics (R-hat, ESS, trace plots) — `az.from_numpyro()` already used, convergence diagnostics not yet wired
- NeuroGym 2.3.0: ContextDecisionMaking-v0 task — Gymnasium-native, no adapter needed
- SciPy + torch.autograd.functional.jacobian: custom fixed point finder — standard Sussillo & Barak 2013 approach, avoids unmaintained libraries
- Pandas 2.0.x + Seaborn 0.13.2: latent circuit output reporting — required by engellab reference code

### Expected Features

**Must have (table stakes) — current milestone:**
- ContextDecisionMaking-v0 task integration — everything in the milestone depends on this environment
- Latent circuit fitting core: Q embedding + w_rec/w_in/w_out optimization with 100-initialization ensemble
- Invariant subspace validation: automated QᵀW_recQ correlation check after each fit
- Activity-level and connectivity-level validation of inferred circuit
- NumPyro MCMC fitting with R-hat, ESS, and trace plot convergence diagnostics
- Parameter recovery simulation (50 synthetic datasets) before fitting real data
- Fit to Nassar 2021 human .mat files (per-subject, per-condition)
- Fixed point finder using input-conditioned dynamics with Jacobian eigendecomposition
- Learning rate comparison: RNN vs schizophrenia patients vs controls

**Should have (differentiators):**
- RNN-human parameter overlay plots: direct visualization of Bayesian parameters across populations
- Latent circuit perturbation analysis: perturb w_rec, predict behavioral change (causal mechanistic claim)
- Bayesian parameter recovery validation with 50-dataset simulation study
- Context error parameters (Nassar 2021 extended 10-parameter model)
- Group-level statistical comparison (schizophrenia vs control) with model information criteria (WAIC)

**Defer (post-MVP):**
- dPCA / demixed PCA of hidden states — high complexity, not required for core comparisons
- Multi-task latent circuit comparison (PIE vs context-DM) — requires stable single-task pipeline first
- Leave-one-subject-out cross-validation — important for clinical classification, defer until basic fits validated
- Clinical correlation with BPRS/SANS/MATRICS scores — requires clinical metadata parsing
- Cross-population latent circuit comparison across training hyperparameters
- GPU-distributed training — current model sizes (hidden_dim=64) do not require it

### Architecture Approach

The pipeline is a five-stage linear DAG with a hard runtime boundary between PyTorch (Stages 1-2) and JAX (Stage 4). The key architectural rule is that all public functions in `analysis/` and `bayesian/` accept and return `np.ndarray`, never `torch.Tensor` or `jnp.ndarray`. Conversion happens inside the module. This keeps callers framework-agnostic and the pipeline script readable. All inter-stage data is persisted as `.npy` files in `data/processed/`, making stages independently re-runnable.

Two structural changes are needed: promote the `FixedPointAnalyzer` class from `scripts/analysis/analyze_fixed_points.py` into `src/nn4psych/analysis/fixed_points.py`, and consolidate `bayesian/numpyro_models.py` into a new `src/nn4psych/bayesian/` subpackage to make the models importable as a proper library. The top-level `bayesian/` directory stays as a deprecated shim.

**Major components:**
1. Stage 1-2: Training + Behavioral Extraction (PyTorch) — existing, outputs `.npy` arrays to `data/processed/rnn_behav/`
2. Stage 3a: Latent Circuit Inference (NumPy/SciPy, new) — `src/nn4psych/analysis/circuit_inference.py`
3. Stage 3b: Fixed Point Analysis (NumPy/SciPy, promote from script) — `src/nn4psych/analysis/fixed_points.py`
4. Stage 4: Bayesian Model Fitting (JAX/NumPyro, consolidate) — `src/nn4psych/bayesian/`
5. Stage 5: Behavioral Comparison (NumPy/Pandas, new) — `src/nn4psych/analysis/comparison.py`
6. Pipeline wiring scripts — `scripts/data_pipeline/08_*.py` through `10_*.py`

### Critical Pitfalls

1. **Single-initialization latent circuit fitting** — only ~10% of random inits converge acceptably; one solution may be an artifact of initialization. Build a 100-init ensemble loop with per-fit QᵀW_recQ validation before any scientific interpretation is attached to a circuit.

2. **JAX `jax.lax.scan` Python string tracing bug** — the existing `bayesian/numpyro_models.py` passes `context` as a Python string into `step_fn` inside `jax.lax.scan`. This silently fixes the branch at trace time, meaning oddball fits may use changepoint equations with no error thrown. Replace with a JAX integer flag using `jnp.where` before any MCMC fitting runs.

3. **GPU memory conflict between JAX and PyTorch** — JAX pre-allocates 75% of GPU VRAM on first use. Set `os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"` and `JAX_PLATFORM_NAME=cpu` before any import of JAX. This must be enforced at the project level, not per-script.

4. **MCMC divergences masked by loose R-hat threshold** — the field standard is R-hat <= 1.01 and zero divergence tolerance. Any divergence triggers reparameterization, not resampling. Enforce this with automated diagnostics; do not rely on visual trace plots alone.

5. **Fixed point search using zero-input dynamics** — the existing `FixedPointAnalyzer` searches `h' = tanh(W_hh @ h)` with no task input. Task-conditioned fixed points occur under constant task inputs; zero-input fixed points may not appear anywhere in actual task trajectories. Revise to pass representative input vectors before any fixed point results are interpreted.

6. **`extract_behavior` uses private environment API** — calls `env._reset_state()` and `env.get_state_history()`, which will break on NeuroGym environments. Refactor to public `env.reset()` / `env.step()` pattern before extending to new tasks.

---

## Implications for Roadmap

Based on the dependency structure in FEATURES.md and the build order in ARCHITECTURE.md, a four-phase structure is recommended. The ordering is dictated by hard data dependencies: latent circuit inference requires RNN hidden states, which require a trained RNN, which requires the context-DM task. Bayesian fitting on RNN outputs requires behavior extraction. Human comparison requires both fitted Bayesian parameters and the human data pipeline.

### Phase 1: Infrastructure Cleanup and Context-DM Task
**Rationale:** Two blocking issues must be resolved before any new capability can be built reliably: the fragile `extract_behavior` private-API dependency (Pitfall 9) and the JAX tracing bug in `numpyro_models.py` (Pitfall 4). The context-DM task must also be integrated first since all latent circuit work depends on a trained context-DM RNN. These are independent of each other and can proceed in parallel.
**Delivers:** Working `extract_behavior` on NeuroGym tasks; `ContextDecisionMaking-v0` integrated and training; JAX tracing bug fixed; pyproject.toml updated; `JAX_PLATFORM_NAME=cpu` enforced project-wide.
**Addresses:** ContextDecisionMaking-v0 task integration, `extract_behavior` refactor
**Avoids:** Pitfalls 3, 4, 9, 13 (infrastructure pitfalls that corrupt all downstream results)
**Research flag:** Standard patterns — no additional research needed. NeuroGym API is documented; JAX fix is known.

### Phase 2: Fixed Point Analysis Module
**Rationale:** Fixed point analysis depends only on a trained RNN and has no dependency on JAX, Bayesian fitting, or latent circuit inference. It can be built and validated immediately after a context-DM RNN is trained. Promoting the existing `FixedPointAnalyzer` class from script to package module is lower risk than building latent circuit inference from scratch, so fixing it first also validates the module promotion pattern before applying it to new code.
**Delivers:** `src/nn4psych/analysis/fixed_points.py` with input-conditioned fixed point search; Jacobian eigendecomposition; PCA visualization of trajectories and fixed points; stability classification including saddle detection.
**Addresses:** Fixed point finder (input-conditioned), Jacobian computation, stability classification, PCA visualization
**Avoids:** Pitfalls 6 (zero-input dynamics), 11 (misaligned PCA spaces), 13 (hardcoded input_dim)
**Research flag:** Standard patterns — Sussillo & Barak 2013 method is well-documented; SciPy L-BFGS-B is established.

### Phase 3: Latent Circuit Inference
**Rationale:** Latent circuit inference is the primary scientific deliverable of the milestone, but it requires trial-averaged hidden states from a trained context-DM RNN and the behavior extraction infrastructure from Phase 1. The multi-initialization requirement (100+ inits, ensemble selection) and the invariant subspace validation must be built into the implementation from the start, not added after results look plausible. Rank selection must be justified by task structure before fitting begins.
**Delivers:** `src/nn4psych/analysis/circuit_inference.py` with 100-init ensemble optimization of Q, w_rec, w_in, w_out; automated QᵀW_recQ correlation diagnostic; activity-level and connectivity-level validation; `scripts/data_pipeline/08_infer_latent_circuits.py`.
**Addresses:** Q/w_rec/w_in/w_out fitting, reconstruction error loss, activity-level validation, connectivity-level validation
**Avoids:** Pitfalls 1 (single-init), 2 (invariant subspace not validated), 7 (arbitrary rank selection), 11 (PCA alignment)
**Research flag:** Needs research-phase. Langdon & Engel 2025 reference implementation details and the specific loss function formulation (Equation 7) should be verified against the engellab/latentcircuit repo during phase planning.

### Phase 4: Bayesian Model Fitting and Human Comparison
**Rationale:** Bayesian fitting depends on behavior extraction (Phase 1 prerequisite) but not on fixed points or latent circuit inference. However, the human comparison (the final scientific output) requires both Bayesian-fitted RNN parameters and human-fitted parameters to be available simultaneously. Consolidating the Bayesian subpackage, validating parameter recovery on synthetic data, fitting human .mat files, and producing the RNN-human comparison are logically coupled and should form a single phase to avoid intermediate states where one fit exists but the comparison cannot yet run.
**Delivers:** `src/nn4psych/bayesian/` subpackage (consolidated from `bayesian/`); parameter recovery simulation (50 synthetic datasets, r >= 0.85 gate); per-subject per-condition MCMC fits with R-hat/ESS/divergence diagnostics; fit to Nassar 2021 .mat files; `src/nn4psych/analysis/comparison.py`; group-level learning rate and parameter comparison; WAIC model comparison; `scripts/data_pipeline/09_*.py` and `10_*.py`.
**Addresses:** NumPyro MCMC fitting, convergence diagnostics, posterior predictive checks, per-subject fits, fit to human data, learning rate comparison, WAIC
**Avoids:** Pitfalls 4 (JAX tracing fixed in Phase 1), 5 (MCMC divergences), 8 (parameter recovery skipped), 12 (fit quality not checked per group), 14 (MAT file structure undocumented), 15 (string args in scan)
**Research flag:** Needs research-phase for Nassar 2021 .mat file structure — the specific nested indexing of the schizophrenia dataset should be documented before data loading code is written.

### Phase Ordering Rationale

- Phase 1 must come first because two bugs (private env API, JAX tracing) will corrupt results from every subsequent phase if not fixed.
- Phase 2 (fixed points) can proceed in parallel with Phase 3 planning since it has no JAX dependency and validates the module-promotion pattern.
- Phase 3 (latent circuits) must follow Phase 1 completion (working extract_behavior on NeuroGym) but can begin as soon as a context-DM RNN is trained.
- Phase 4 (Bayesian + comparison) depends on Phase 1 infrastructure but is otherwise independent of Phases 2 and 3, allowing it to run in parallel once the behavior extraction pipeline is validated.
- Perturbation analysis (causal mechanistic claim) is deliberately excluded from the roadmap as a separate post-phase deliverable — it requires validated connectivity-level fits from Phase 3 and is high enough complexity to warrant its own milestone.

### Research Flags

Phases likely needing `/gsd:research-phase` during planning:
- **Phase 3 (Latent Circuit Inference):** The Langdon & Engel loss function formulation, the specific parameterization of Q (orthonormality constraint), and the ensemble selection criterion are described in the paper but must be verified against the engellab/latentcircuit reference code before implementation to avoid reimplementing incorrectly.
- **Phase 4 (Bayesian + Human Comparison):** The Nassar 2021 .mat file structure is hardcoded and undocumented in the existing pipeline. The specific nested indexing must be inspected from the actual data files before any loading code is written.

Phases with standard patterns (skip additional research):
- **Phase 1 (Infrastructure):** NeuroGym API, JAX platform config, PyTorch public gym API — all well-documented.
- **Phase 2 (Fixed Points):** Sussillo & Barak 2013 method is standard; SciPy L-BFGS-B + torch.autograd.functional.jacobian pattern is well-established.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All packages verified against PyPI releases and official repos; version constraints confirmed; JAX/PyTorch coexistence strategy verified from official JAX docs and existing codebase patterns |
| Features | MEDIUM-HIGH | Core features verified against Langdon & Engel 2025 (full PMC text), Nassar 2021 (full PMC text), FixedPointFinder JOSS paper; ContextDecisionMaking-v0 docs are incomplete (MEDIUM) |
| Architecture | HIGH | Build order derived from direct dependency analysis of existing codebase files; NumPy handoff pattern confirmed in existing `run_mcmc` and `posterior_predictive` implementations |
| Pitfalls | HIGH (critical), MEDIUM (moderate) | Critical pitfalls verified from primary paper + official JAX docs + direct code inspection; bug at numpyro_models.py line 149 confirmed by code review; moderate pitfalls inferred from codebase inspection with MEDIUM confidence |

**Overall confidence:** HIGH for implementation decisions; MEDIUM for the Nassar .mat data structure (not directly inspected).

### Gaps to Address

- **Nassar .mat file structure:** The exact nested indexing of the schizophrenia dataset has not been directly inspected — the existing hardcoded indexing in `06_compare_with_human_data.py` may or may not match the actual files. Must be resolved at the start of Phase 4 with a `describe_mat_structure()` inspection utility before any data loading code is written.
- **Latent circuit rank selection for context-DM task:** Langdon & Engel chose n=8 for their specific RNN configuration. The correct rank for the nn4psych RNN (which may differ in hidden_dim and training regime) must be determined during Phase 3 using task structure analysis and behavioral output loss as validation.
- **FixedPointFinder PyTorch 2.x compatibility:** The mattgolub implementation has broken tests since 2022 — the custom implementation approach is recommended but its output should be cross-validated against the FixedPointFinder reference outputs on a small test case to ensure correctness.
- **Optax / SVI:** If NUTS sampling proves too slow for hyperparameter sweeps at scale (3,456 model variants), switching to SVI (variational inference) via Optax is the natural next step. The decision point is not yet reached; defer but keep Optax in mind.

---

## Sources

### Primary (HIGH confidence)
- Langdon & Engel 2025, Nature Neuroscience / PMC11893458 — latent circuit inference method, feature requirements, multi-init requirement, invariant subspace condition
- engellab/latentcircuit GitHub — reference PyTorch implementation, dependency versions
- Nassar 2021, PMC8041039 — Bayesian cognitive model structure, parameter recovery standards, human behavioral data format
- NumPyro 0.20.0 PyPI / official docs — MCMC API, NUTS, Predictive
- JAX 0.9.1 PyPI / official GPU memory docs — platform config, pre-allocation behavior
- ArviZ 1.0.0 PyPI — MCMC diagnostics API
- NeuroGym 2.3.0 GitHub / docs — ContextDecisionMaking-v0 task specification
- Existing codebase (direct inspection): `bayesian/numpyro_models.py`, `scripts/analysis/analyze_fixed_points.py`, `src/nn4psych/analysis/behavior.py`, `scripts/training/train_multitask.py`

### Secondary (MEDIUM confidence)
- JAX GitHub issues #8362, #15084, #15268, #19213 — GPU memory conflict behavior
- Golub & Sussillo FixedPointFinder JOSS paper / GitHub — fixed point methodology (tests broken, not used as dependency)
- "Troubleshooting Bayesian cognitive models" PMC10522800 — MCMC divergence standards
- torch_jax_interop (mila-iqia) — DLPack approach (not used, documented as future option)
- Pals et al. 2024, NeurIPS — stochastic low-rank RNN inference (context, not adopted)

### Tertiary (LOW confidence)
- Valente et al. 2022/2024 — variational latent circuit approach (excluded from scope as anti-feature)
- Frontiers psychiatry review on computational models in psychosis — domain background only

---
*Research completed: 2026-03-18*
*Ready for roadmap: yes*
