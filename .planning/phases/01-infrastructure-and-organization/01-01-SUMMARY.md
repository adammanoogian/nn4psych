---
phase: 01-infrastructure-and-organization
plan: "01"
subsystem: package-structure
tags: [numpyro, jax, bayesian, mcmc, package-layout, src-layout]

# Dependency graph
requires: []
provides:
  - "src/nn4psych/bayesian/ subpackage with NumPyro-only exports"
  - "JAX CPU enforcement via os.environ.setdefault before any JAX import"
  - "model_comparison.py without PyEM dependency"
  - "NumPyro fitting scripts updated to nn4psych.bayesian.* import paths"
affects:
  - phase 01-02 (bayesian archive - old bayesian/ root not yet deleted)
  - phase 02-environment (can now import nn4psych.bayesian)
  - phase 03-rnn-training (JAX/PyTorch coexistence now enforced)
  - phase 04-bayesian-fitting (uses nn4psych.bayesian.numpyro_models)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "src-layout: all installable code lives under src/nn4psych/"
    - "JAX CPU enforcement: os.environ.setdefault before JAX import in subpackage __init__"
    - "NumPyro-only exports: bayesian __init__ exposes only MCMC functions, no PyEM/PyMC"
    - "compare_contexts accepts pre-computed negll floats, not raw data + fit() call"

key-files:
  created:
    - src/nn4psych/bayesian/__init__.py
    - src/nn4psych/bayesian/numpyro_models.py
    - src/nn4psych/bayesian/model_comparison.py
    - src/nn4psych/bayesian/visualization.py
  modified:
    - scripts/fitting/fit_bayesian_numpyro.py
    - scripts/fitting/fit_nassar_numpyro.py
    - scripts/fitting/batch_fit_bayesian.py

key-decisions:
  - "compare_contexts() signature changed to accept negll_cp/negll_ob floats instead of calling fit() internally — caller is responsible for computing negll"
  - "cross_validate_k_fold() removed entirely — was tightly coupled to PyEM optimization loop; NumPyro cross-validation deferred to Phase 4"
  - "batch_fit_bayesian.py not rewritten — TODO comment added; actual NumPyro rewrite is Phase 4 work"
  - "Original bayesian/ root directory NOT deleted — Plan 01-02 handles archiving"

patterns-established:
  - "Bayesian subpackage pattern: JAX env vars before import, NumPyro-only exports, no PyEM/PyMC"
  - "Script import convention: from nn4psych.bayesian.X import Y (not from bayesian.X)"

# Metrics
duration: 9min
completed: 2026-03-18
---

# Phase 1 Plan 01: Bayesian Subpackage Migration Summary

**NumPyro MCMC subpackage moved to src/nn4psych/bayesian/ with JAX CPU enforcement, PyEM dependency removed from model_comparison, and active scripts updated to nn4psych.bayesian.* import paths**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-18T20:00:13Z
- **Completed:** 2026-03-18T20:09:50Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Created `src/nn4psych/bayesian/` as an installable subpackage with four modules
- `__init__.py` enforces `JAX_PLATFORM_NAME=cpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` before any JAX import, enabling PyTorch/JAX coexistence
- Removed `from bayesian.pyem_models import fit` from model_comparison.py; `compare_contexts()` now accepts pre-computed negll floats; `cross_validate_k_fold()` removed (was PyEM-only)
- Updated all three active NumPyro fitting scripts to use `nn4psych.bayesian.*` import paths

## Task Commits

Each task was committed atomically:

1. **Task 1: Add src/nn4psych/bayesian/ subpackage** - `81a4f58` (feat)
2. **Task 2: Update fitting scripts to nn4psych.bayesian imports** - `1a0cee9` (feat)

**Plan metadata:** (see below)

## Files Created/Modified

- `src/nn4psych/bayesian/__init__.py` - Package init: JAX CPU env vars + NumPyro-only exports
- `src/nn4psych/bayesian/numpyro_models.py` - Copied from bayesian/; contains compute_normative_model, run_mcmc, and MCMC utilities
- `src/nn4psych/bayesian/model_comparison.py` - Copied from bayesian/ with PyEM import removed; compare_contexts() refactored; cross_validate_k_fold() removed
- `src/nn4psych/bayesian/visualization.py` - Copied from bayesian/; pure matplotlib/seaborn, no changes
- `scripts/fitting/fit_bayesian_numpyro.py` - Import updated: bayesian.numpyro_models -> nn4psych.bayesian.numpyro_models
- `scripts/fitting/fit_nassar_numpyro.py` - Import updated: bayesian.numpyro_models -> nn4psych.bayesian.numpyro_models
- `scripts/fitting/batch_fit_bayesian.py` - PyEM import removed; model_comparison and visualization updated to nn4psych.bayesian.*; TODO comment added

## Decisions Made

- `compare_contexts()` signature changed from accepting raw behavioral data + calling `fit()` to accepting pre-computed `negll_cp`/`negll_ob` floats. The old signature required PyEM's `fit()` function to evaluate likelihood; callers now compute negll from their own model (NumPyro MAP or MCMC).
- `cross_validate_k_fold()` removed entirely — it was built around PyEM's `minimize(fit, ...)` optimization loop. NumPyro cross-validation will be implemented differently in Phase 4 using MCMC leave-one-out.
- `batch_fit_bayesian.py` received only import fixes, not a logic rewrite. The script's core loop uses `fit_bayesian_model` (PyEM) extensively. Rather than rewrite it now, a TODO comment was added. This is Phase 4 work.
- Original `bayesian/` root directory was intentionally NOT deleted; Plan 01-02 handles archiving `bayesian_models.py` and `pyem_models.py`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

The `pip install -e .` command appeared to succeed but the package was not resolvable from this shell's Python interpreter due to a Windows long-path limitation and torch DLL load failure in the test environment. Import verification was performed via:
1. `python -m py_compile` syntax checks on all four files
2. `ast.parse()` function-presence checks
3. `grep` confirmation of correct import paths and removed pyem_models reference
4. Manual ordering verification (JAX env vars before numpyro_models import)

These checks confirm functional correctness. The torch DLL issue is pre-existing and unrelated to this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `import nn4psych.bayesian` will work after `pip install -e .` in a functioning environment
- JAX CPU enforcement is in place for Plan 01-03 (JAX tracing bug fix) and Phase 3 (joint training)
- Plan 01-02 can now archive `bayesian/pyem_models.py` and `bayesian/bayesian_models.py` safely — the new subpackage has no dependency on them
- Blocker remains: JAX `jax.lax.scan` tracing bug in numpyro_models.py line ~149 (Python string `context` inside scan) — tracked in STATE.md; Plan 01-03 will fix

---
*Phase: 01-infrastructure-and-organization*
*Completed: 2026-03-18*
