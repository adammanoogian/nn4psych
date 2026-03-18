---
phase: 01-infrastructure-and-organization
plan: 02
subsystem: infra
tags: [pyproject-toml, jax, numpyro, arviz, bayesian, pymc, archive, python311, ruff, mypy]

# Dependency graph
requires: []
provides:
  - pyproject.toml with [bayesian] extra pointing to JAX/NumPyro/ArviZ (no PyMC/PyTensor)
  - Python >=3.11 requirement enforced
  - Dev tool pins updated to project_utils conventions
  - archive/bayesian_pymc/ containing all PyMC and PyEM implementation files
affects:
  - 01-03-PLAN (final cleanup of bayesian/ directory)
  - Phase 3 (Bayesian model fitting: depends on [bayesian] extra resolving correctly)

# Tech tracking
tech-stack:
  added: [jax>=0.4.0, jaxlib>=0.4.0, numpyro>=0.13.0, arviz>=0.17.0]
  patterns:
    - "Single optional extras group [bayesian] for all probabilistic computing dependencies"
    - "Archived deprecated implementations to archive/{subsystem}/ with README"

key-files:
  created:
    - archive/bayesian_pymc/README.md
    - archive/bayesian_pymc/bayesian_models.py
    - archive/bayesian_pymc/pyem_models.py
    - archive/bayesian_pymc/fit_bayesian_pymc.py
    - archive/bayesian_pymc/fit_bayesian_pyem.py
  modified:
    - pyproject.toml

key-decisions:
  - "Merged [jax] extra into [bayesian] — no standalone JAX use case in this project"
  - "Dev pins bumped to pytest>=8.0, ruff>=0.4, mypy>=1.10 per project_utils conventions"
  - "Archived fitting scripts alongside model files so archive is self-contained"
  - "Original bayesian/ directory left intact — cleanup deferred to plan 01-03"

patterns-established:
  - "archive/{subsystem}/ contains all deprecated files with a README explaining provenance"
  - "Ruff per-file-ignores for tests/ (S101, D) and scripts/ (D, T201) suppresses expected lint violations"

# Metrics
duration: 4min
completed: 2026-03-18
---

# Phase 1 Plan 02: Dependency Configuration and PyMC Archive Summary

**pyproject.toml updated to JAX/NumPyro/ArviZ [bayesian] extra with Python >=3.11; PyMC and PyEM implementations archived to archive/bayesian_pymc/**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-18T20:00:22Z
- **Completed:** 2026-03-18T20:04:09Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Replaced [bayesian] optional dependency group: removed pymc>=4.0.0 and pytensor>=2.0.0, added jax>=0.4.0, jaxlib>=0.4.0, numpyro>=0.13.0, arviz>=0.17.0
- Removed standalone [jax] extra (JAX is now accessible only through [bayesian])
- Bumped requires-python to >=3.11; classifiers updated to reflect 3.11 and 3.12
- Updated dev tool pins to project_utils conventions: pytest>=8.0, pytest-cov>=5.0, ruff>=0.4, mypy>=1.10
- Added ruff format config, per-file-ignores, and pytest filterwarnings
- Bumped build-system to setuptools>=68.0 and setuptools-scm>=8.0
- Archived bayesian_models.py, pyem_models.py, fit_bayesian_pymc.py, fit_bayesian_pyem.py to archive/bayesian_pymc/ with README

## Task Commits

Each task was committed atomically:

1. **Task 1: Update pyproject.toml dependencies, Python version, and tool config** - `26c7509` (chore)
2. **Task 2: Archive PyMC and PyEM model files** - `b4df3ca` (chore)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `pyproject.toml` - Updated [bayesian] extra (JAX/NumPyro/ArviZ), removed [jax] extra, bumped Python to >=3.11, updated dev pins, added ruff format/ignores, filterwarnings, build-system versions
- `archive/bayesian_pymc/bayesian_models.py` - Archived PyMC-based MCMC model
- `archive/bayesian_pymc/pyem_models.py` - Archived PyEM-based MLE/MAP model
- `archive/bayesian_pymc/fit_bayesian_pymc.py` - Archived PyMC fitting script
- `archive/bayesian_pymc/fit_bayesian_pyem.py` - Archived PyEM fitting script
- `archive/bayesian_pymc/README.md` - Archive provenance and restoration instructions

## Decisions Made

- Merged [jax] extra into [bayesian]: there is no standalone JAX use case in this project; all JAX usage is in service of NumPyro/ArviZ probabilistic computing
- Dev pin versions aligned to project_utils conventions (pytest>=8.0, ruff>=0.4, mypy>=1.10) to match standard tooling expectations
- Fitting scripts archived alongside model files so archive/bayesian_pymc/ is self-contained and could be restored without cross-referencing scripts/ tree
- Original bayesian/ directory NOT deleted — that cleanup is deferred to plan 01-03 to avoid conflicts with any parallel plan 01-01 work

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `pip install -e ".[bayesian]"` will now install JAX, NumPyro, and ArviZ — no PyMC or PyTensor
- Cleanup of the original bayesian/ directory (remove bayesian_models.py, pyem_models.py from src) is deferred to plan 01-03
- Known blocker from STATE.md remains: JAX tracing bug in numpyro_models.py (line ~149) — Python string `context` inside jax.lax.scan silently ignores oddball condition; must fix before any Bayesian fitting in Phase 3

---
*Phase: 01-infrastructure-and-organization*
*Completed: 2026-03-18*
