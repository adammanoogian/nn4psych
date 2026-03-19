# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-18)

**Core value:** RNN agent trainable on multiple cognitive tasks with analyzable hidden representations comparable to human data via Bayesian model fitting
**Current focus:** Phase 2 — RNN Training Verification

## Current Position

Phase: 2 of 5 (RNN Training Verification) — In progress
Plan: 2 of 3 in phase 02 (02-01, 02-02 complete)
Status: In progress — Phase 2 Plan 2 complete
Last activity: 2026-03-19 — Completed 02-02-PLAN.md (extract_behavior_with_hidden, end-to-end training verification)

Progress: [█████░░░░░] ~38% (5/~13 total plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: ~10 min
- Total execution time: ~0.65 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-infrastructure-and-organization | 3/3 COMPLETE | ~24 min | ~8 min |
| 02-rnn-training-verification | 2/3 | ~27 min | ~13 min |

**Recent Trend:**
- Last 5 plans: 01-01 (9 min), 01-02 (unknown), 01-03 (7 min), 02-01 (12 min), 02-02 (15 min)
- Trend: ~7-15 min per plan

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: NumPyro/JAX chosen over PyMC for Bayesian models (JAX faster, composable)
- [Init]: PyEM models to be archived, not maintained
- [Init]: Latent circuit inference (Langdon & Engel 2025) chosen over standard dim reduction
- [Init]: Context-DM task required — primary task for latent circuit analysis
- [01-01]: compare_contexts() now accepts pre-computed negll floats — caller computes negll from own model
- [01-01]: cross_validate_k_fold() removed — was PyEM-only; NumPyro CV deferred to Phase 4
- [01-01]: batch_fit_bayesian.py not rewritten; TODO comment added; Phase 4 work
- [01-01]: Original bayesian/ root NOT deleted — Plan 01-02 handles archiving
- [01-03]: jax.lax.cond with operand=None and lambda _: pattern for closed-over variable branches
- [01-03]: is_changepoint = jnp.bool_(context == 'changepoint') outside step_fn — JAX tracer-compatible
- [01-03]: NeurogymWrapper.reset_epoch() resets self.trial = 0 plus all history lists
- [01-03]: env_params defaults to {} inside batch_extract_behavior (not as default arg) to avoid mutable default anti-pattern
- [02-01]: evaluate() method also needs hasattr guard — 4 sites total not 3; auto-fixed as Rule 1 bug
- [02-01]: obs_dim for ContextDecisionMaking is 1 + 2*dim_ring (not 1 + dim_ring); fixation + 2 modalities x ring_units
- [02-01]: neurogym installed in actinf-py-scripts conda env (v2.2.0); is the project's working Python environment
- [02-02]: extract_behavior_with_hidden uses argmax (deterministic) not sampling — consistent with extract_behavior(); reproducible for analysis
- [02-02]: NaN padding for hidden states (not zeros) so downstream masking can identify real zero activations vs padded timesteps
- [02-02]: DawTwoStep NeurogymWrapper produces max_T~1000 (long trials at dt=100ms) — may need trial length filtering before latent circuit fitting in Phase 3

### Pending Todos

None.

### Blockers/Concerns

- [RESOLVED 01-03]: JAX tracing bug in numpyro_models.py — FIXED with jax.lax.cond
- [RESOLVED 01-03]: extract_behavior private env API — FIXED with reset_epoch() public method
- [Phase 3 planning]: Latent circuit rank selection for context-DM task needs verification against engellab/latentcircuit repo (research flag)
- [Phase 4 planning]: Nassar 2021 .mat file nested indexing not directly inspected — must run describe_mat_structure() before writing data loading code (research flag)

## Session Continuity

Last session: 2026-03-19T12:29:00Z
Stopped at: Completed 02-02-PLAN.md — Phase 2, Plan 2 complete
Resume file: None
