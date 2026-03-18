---
phase: 01-infrastructure-and-organization
plan: "03"
subsystem: bayesian, envs, analysis
tags: [jax, jax.lax.cond, numpyro, normative-model, pie-environment, neurogym, behavior-extraction]

# Dependency graph
requires:
  - phase: 01-01
    provides: src/nn4psych/bayesian/ subpackage with numpyro_models.py at new location
provides:
  - JAX-traced normative model using jax.lax.cond for context branching (bug fixed)
  - Public reset_epoch() on PIE_CP_OB_v2 and NeurogymWrapper
  - extract_behavior using public env API (reset_epoch, get_state_history)
  - batch_extract_behavior with parameterized input_dim, hidden_dim, action_dim, env_params
affects:
  - Phase 2 (Bayesian fitting): normative model now correctly differentiates contexts
  - Phase 2 (behavior extraction): extract_behavior now works on NeuroGym environments
  - Any script using batch_extract_behavior with non-default model dimensions

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "jax.lax.cond for JAX-compatible conditionals inside jax.lax.scan step functions"
    - "is_changepoint = jnp.bool_(context == 'changepoint') outside step_fn, closed over"
    - "reset_epoch() public method delegating to private _reset_state() in PIE env"
    - "reset_epoch() on NeurogymWrapper resets trial, trials, rewards/actions/observations history"

key-files:
  created: []
  modified:
    - src/nn4psych/bayesian/numpyro_models.py
    - src/nn4psych/analysis/behavior.py
    - envs/pie_environment.py
    - envs/neurogym_wrapper.py

key-decisions:
  - "jax.lax.cond with operand=None and lambda _: pattern is standard for closed-over variables"
  - "NeurogymWrapper.reset_epoch() mirrors _reset_history() plus resets self.trial counter"
  - "extract_behavior env type annotation broadened to plain comment (no Protocol) for simplicity"
  - "env_params defaults to {} in batch_extract_behavior to preserve backward compatibility"

patterns-established:
  - "JAX-compatible conditionals: always jax.lax.cond, never Python if/else inside scan step_fn"
  - "Public env API: reset_epoch() for epoch boundary, get_state_history() for data access"

# Metrics
duration: 7min
completed: 2026-03-18
---

# Phase 1 Plan 3: Bug Fixes — JAX Tracing, Public Env API, Parameterized Dims Summary

**jax.lax.cond replaces Python if/else in normative model scan, reset_epoch() public API on both envs, behavior extraction fully decoupled from private env internals**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-03-18T20:12:17Z
- **Completed:** 2026-03-18T20:19:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Fixed critical JAX tracing bug: `compute_normative_model` now produces verifiably different learning rates for changepoint vs. oddball contexts at JAX runtime (not just trace time)
- Added `reset_epoch()` public method to both `PIE_CP_OB_v2` and `NeurogymWrapper`, providing a stable public interface for epoch boundary resets
- Fixed `extract_behavior` to call `env.reset_epoch()` instead of private `env._reset_state()`, enabling use with NeuroGym environments
- Parameterized `batch_extract_behavior` with `input_dim`, `hidden_dim`, `action_dim`, `env_params` — eliminating hardcoded 9/64/3 defaults that would silently fail on different architectures

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix JAX tracing bug in compute_normative_model** - `65b0c5a` (fix)
2. **Task 2: Add reset_epoch() to environments and fix extract_behavior** - `d6a52b9` (fix)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `src/nn4psych/bayesian/numpyro_models.py` - Replaced Python if/else with jax.lax.cond; removed pre-allocated unused arrays; added is_changepoint computed outside step_fn
- `envs/pie_environment.py` - Added reset_epoch() delegating to _reset_state()
- `envs/neurogym_wrapper.py` - Added reset_epoch() resetting trial, trials, rewards/actions/observations_history, trial_lengths
- `src/nn4psych/analysis/behavior.py` - Changed env._reset_state() to env.reset_epoch(); broadened env type hint; parameterized batch_extract_behavior dimensions

## Decisions Made
- Used `jax.lax.cond(pred, lambda _: ..., lambda _: ..., operand=None)` pattern — standard when branches use only closed-over variables, avoids needing to pass operand
- `is_changepoint = jnp.bool_(context == 'changepoint')` placed before `step_fn` definition so it is a JAX-tracer-compatible closed-over value, not evaluated per-scan-iteration
- Removed pre-allocated `learning_rate`, `omega`, `tau`, `normative_update` arrays (lines 104-107 in original) — they were allocated but never used since jax.lax.scan returns outputs directly; removing them avoids confusion
- `NeurogymWrapper.reset_epoch()` resets `self.trial = 0` in addition to all history lists — trial counter must reset or epoch boundary isn't clean
- `env_params` defaults to `{}` inside `batch_extract_behavior` (not as default argument) to avoid mutable default argument Python anti-pattern

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Python 3.13 system default lacks torch DLLs, making `from nn4psych.bayesian.numpyro_models import ...` fail via package `__init__.py` (which imports torch). Worked around verification by importing the module directly via `importlib.util.spec_from_file_location` and using the `ds_env` conda environment (has JAX, no torch needed for normative model verification).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 1 complete: all three blocking bugs resolved
- Normative model ready for Bayesian fitting (Phase 2): contexts produce distinct learning rates
- Behavior extraction ready for NeuroGym environments (Phase 2): public API, not private
- All Phase 1 blockers/concerns from STATE.md are now resolved:
  - [RESOLVED] JAX tracing bug in numpyro_models.py
  - [RESOLVED] extract_behavior private env API (_reset_state)
- Remaining concerns carry to Phase 2:
  - [Phase 3 planning]: Latent circuit rank selection verification against engellab repo
  - [Phase 4 planning]: Nassar 2021 .mat file nested indexing not directly inspected

---
*Phase: 01-infrastructure-and-organization*
*Completed: 2026-03-18*
