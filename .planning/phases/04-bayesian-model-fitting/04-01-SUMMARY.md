---
phase: 04-bayesian-model-fitting
plan: "01"
subsystem: bayesian
tags: [numpyro, jax, mcmc, nuts, arviz, reduced-bayesian-observer, nassar2021]

# Dependency graph
requires:
  - phase: 01-infrastructure-and-organization
    provides: "JAX-tracer-compatible jax.lax.cond + jnp.bool_ pattern (01-03 lesson); bayesian/ package scaffold"
provides:
  - "reduced_bayesian.py: canonical NumPyro/JAX Reduced Bayesian Observer with paper-informed priors"
  - "run_mcmc: Phase 4 defaults (4 chains x 2000 warmup x 2000 draws, extra_fields=('diverging',))"
  - "prior_sampler, simulate_synthetic_data, assert_jax_devices helpers"
  - "XLA_FLAGS set in __init__.py for 4 virtual CPU devices before any jax import"
  - "BAYES-01 closed: PyEM/PyMC archived; cluster/batch_fit_bayesian.py is fail-fast stub"
  - "6 smoke tests in tests/test_reduced_bayesian.py"
  - "arviz pinned to >=0.17.0,<0.25.0"
affects:
  - "04-02: imports reduced_bayesian_model, prior_sampler, simulate_synthetic_data, run_mcmc from nn4psych.bayesian"
  - "04-03: imports run_mcmc; calls Brain2021Code data fetch which gates prior verification"
  - "04-04b: imports run_mcmc for RNN cohort fitting"
  - "All Phase 4 plans: Predictive must be imported directly from numpyro.infer (not re-exported here)"

# Tech tracking
tech-stack:
  added: [pytest-cov (installed to unblock pytest runner; was pre-existing missing dep)]
  patterns:
    - "is_changepoint = jnp.bool_(context == 'changepoint') outside step_fn (01-03 Phase 1 JAX-tracer pattern)"
    - "XLA_FLAGS set via os.environ.setdefault() before any jax import in __init__.py"
    - "FALLBACK prior annotation pattern on all numpyro.sample() calls (pending supplement verification)"
    - "extra_fields=('diverging',) required kwarg to mcmc.run() for 04-02 divergence diagnostics"

key-files:
  created:
    - src/nn4psych/bayesian/reduced_bayesian.py
    - tests/test_reduced_bayesian.py
  modified:
    - src/nn4psych/bayesian/__init__.py
    - src/nn4psych/bayesian/numpyro_models.py
    - cluster/batch_fit_bayesian.py
    - archive/bayesian_legacy/README.md
    - archive/bayesian_pymc/README.md
    - pyproject.toml

key-decisions:
  - "All five priors (H, LW, UU, sigma_motor, sigma_LR) are FALLBACK pending Nassar 2021 supplement download in Plan 04-03 Task 1"
  - "Predictive NOT re-exported from nn4psych.bayesian (m9 fix); downstream callers import from numpyro.infer directly"
  - "numpyro_models.py deprecated but not deleted (git-history continuity for Phase 1 SUMMARY references)"
  - "compute_rbo_forward reuses full predictive-variance-weighted tau update from numpyro_models.py lines 133-138 (not simplified tau/UU form from metrics.py)"
  - "run_mcmc signature: bag_positions FIRST, bucket_positions SECOND (differs from numpyro_models.py which had bucket first)"
  - "arviz pinned >=0.17.0,<0.25.0 to prevent unexpected API breaks (RESEARCH.md Open Question 5)"

patterns-established:
  - "Phase 4 canonical import: from nn4psych.bayesian import reduced_bayesian_model, run_mcmc, prior_sampler"
  - "NumPyro prior annotation format: # FALLBACK pending ... / # PAPER_VERIFIED (for future supplement confirmation)"

# Metrics
duration: 15min
completed: 2026-04-30
---

# Phase 4 Plan 01: Reduced Bayesian Observer Implementation Summary

**NumPyro/JAX Reduced Bayesian Observer (Nassar 2010+2021) with 5-param weakly-informative priors, 4-CPU-chain NUTS defaults, BAYES-01 archive closure, and 6 smoke tests**

## Performance

- **Duration:** 15 min
- **Started:** 2026-04-30T06:00:19Z
- **Completed:** 2026-04-30T06:15:21Z
- **Tasks:** 3 / 3
- **Files modified:** 8

## Accomplishments

- Created `src/nn4psych/bayesian/reduced_bayesian.py` — canonical Phase 4 forward model implementing Nassar 2010+2021 Reduced Bayesian Observer with JAX-scan + jax.lax.cond pattern, paper-traceable prior annotations, and Phase 4 MCMC defaults
- Closed BAYES-01: PyEM/PyMC archive READMEs updated; `cluster/batch_fit_bayesian.py` replaced with fail-fast stub; no stale symbols importable from `nn4psych.bayesian`
- 4-CPU-chain support via `XLA_FLAGS` set in `__init__.py` before any JAX import; `jax.local_device_count() == 4` verified

## Task Commits

Each task was committed atomically:

1. **Task 1: Create reduced_bayesian.py with corrected forward model and documented priors** - `4400611` (feat)
2. **Task 2: Update __init__.py with XLA_FLAGS and new exports; deprecate numpyro_models.py** - `2f50018` (feat)
3. **Task 3: Close BAYES-01 (archive READMEs, cluster stub), add smoke tests, pin arviz** - `7b0cd7d` (chore)

## Files Created/Modified

- `src/nn4psych/bayesian/reduced_bayesian.py` — NEW: canonical RBO forward model (`compute_rbo_forward`), NumPyro model (`reduced_bayesian_model`), MCMC entry point (`run_mcmc`), helpers (`prior_sampler`, `simulate_synthetic_data`, `assert_jax_devices`)
- `src/nn4psych/bayesian/__init__.py` — UPDATED: XLA_FLAGS for 4 virtual CPU devices; exports Phase 4 canonical surface only; stale numpyro_models.py symbols removed
- `src/nn4psych/bayesian/numpyro_models.py` — UPDATED: DeprecationWarning at import; `from __future__ import annotations`; M2 typing cleanup (Dict/Optional/Tuple → native 3.10+ types)
- `cluster/batch_fit_bayesian.py` — REPLACED: fail-fast stub (exits 2) pointing to 09/10 data pipeline scripts
- `archive/bayesian_legacy/README.md` — UPDATED: BAYES-01 section appended
- `archive/bayesian_pymc/README.md` — UPDATED: BAYES-01 section appended
- `tests/test_reduced_bayesian.py` — NEW: 6 smoke tests (forward model JIT x2, prior shapes, synthetic data shapes, MCMC smoke, device check)
- `pyproject.toml` — UPDATED: arviz pin changed from `>=0.17.0` to `>=0.17.0,<0.25.0`

## Decisions Made

1. **All five priors are FALLBACK**: `H ~ Beta(1.5, 8)`, `LW ~ Beta(2, 2)`, `UU ~ HalfNormal(0.5)`, `sigma_motor ~ HalfNormal(10.0)`, `sigma_LR ~ HalfNormal(1.0)`. These are weakly-informative defaults. The paper specifies "informed prior derived from MLE fits" — exact distributional form is in the Nassar 2021 supplement, which requires downloading Brain2021Code (Plan 04-03 Task 1 gates this). Each prior has an inline `# FALLBACK pending Nassar 2021 supplement` comment.

2. **Predictive NOT re-exported** (m9 fix): `nn4psych.bayesian.__all__` does not include `Predictive`. Downstream plans import `from numpyro.infer import Predictive` directly. This avoids leaking NumPyro internals through the project namespace.

3. **numpyro_models.py deprecated but retained**: Deleted would break Phase 1 SUMMARY git-history references. Deprecation warning fires at import; all typing cleaned to Python 3.10+ native syntax.

4. **Full tau update retained**: `compute_rbo_forward` uses the full predictive-variance-weighted tau update from `numpyro_models.py` lines 133-138, not the simplified `tau / UU` form in `metrics.py`. Flagged in code comments for validation in 04-02 against Nassar MATLAB reference.

5. **run_mcmc argument order**: `bag_positions` first, `bucket_positions` second (semantic clarity: bag is the generative input, bucket is the agent response). This differs from `numpyro_models.py` which had bucket first.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing pytest-cov dependency**
- **Found during:** Task 3 (smoke test verification)
- **Issue:** `pytest.ini` had `--cov=nn4psych` in `addopts` but `pytest-cov` was not installed in the conda env; pytest exited with error code 4 on any invocation
- **Fix:** `pip install pytest-cov` in the actinf-py-scripts env
- **Files modified:** None (env-only change)
- **Verification:** pytest ran successfully and all 6 tests passed
- **Committed in:** not staged (env install only; not a source change)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** pytest-cov install was purely a blocking infrastructure fix. No scope creep.

## Issues Encountered

- Python 3.10 environment lacks `tomllib` (added in 3.11 stdlib). Used `tomli` (already installed) to verify TOML syntax. The project's `pyproject.toml` specifies `requires-python = ">=3.11"` but the active conda env is 3.10 — this is a pre-existing environment mismatch, not introduced by this plan.

## Next Phase Readiness

**Ready for 04-02:**
- `nn4psych.bayesian.reduced_bayesian_model`, `run_mcmc`, `prior_sampler`, `simulate_synthetic_data` all importable and tested
- `extra_fields=('diverging',)` plumbing verified: `mcmc.get_extra_fields()['diverging']` accessible
- 4-chain parallel NUTS working: `jax.local_device_count() == 4`

**Open items (not blocking 04-02):**
- Prior verification against Nassar 2021 supplement — gated on Plan 04-03 Task 1 Brain2021Code download
- Tau update equation correctness vs Nassar MATLAB reference — flagged for 04-02 parameter recovery validation
- If parameter recovery in 04-02 fails (r < 0.85 for any parameter), first diagnostic is tau equation form

**Blocker for 04-03:**
- Raw Nassar 2021 behavioral data (`realSubjects/` directory) is not on this machine. Plan 04-03 Task 1 must download Brain2021Code from sites.brown.edu/mattlab/resources/ before any fitting pipeline can run.

---
*Phase: 04-bayesian-model-fitting*
*Completed: 2026-04-30*
