# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-18)

**Core value:** RNN agent trainable on multiple cognitive tasks with analyzable hidden representations comparable to human data via Bayesian model fitting
**Current focus:** Phase 1 — Infrastructure and Organization

## Current Position

Phase: 1 of 5 (Infrastructure and Organization)
Plan: 1 of 3 in current phase (01-02 complete)
Status: In progress
Last activity: 2026-03-18 — Completed 01-02-PLAN.md (pyproject.toml + PyMC archive)

Progress: [█░░░░░░░░░] ~7% (1/~13 total plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 4 min
- Total execution time: ~0.07 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-infrastructure-and-organization | 1/3 | 4 min | 4 min |

**Recent Trend:**
- Last 5 plans: 01-02 (4 min)
- Trend: —

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: NumPyro/JAX chosen over PyMC for Bayesian models (JAX faster, composable)
- [Init]: PyEM models to be archived, not maintained
- [Init]: Latent circuit inference (Langdon & Engel 2025) chosen over standard dim reduction
- [Init]: Context-DM task required — primary task for latent circuit analysis
- [01-02]: Merged [jax] extra into [bayesian] — no standalone JAX use case in this project
- [01-02]: Fitting scripts archived alongside model files so archive/bayesian_pymc/ is self-contained
- [01-02]: bayesian/ directory cleanup deferred to plan 01-03 (after plan 01-01 completes)

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1 prereq]: JAX tracing bug in numpyro_models.py (line ~149) — Python string `context` inside jax.lax.scan silently ignores oddball condition; must fix before any Bayesian fitting
- [Phase 1 prereq]: extract_behavior uses private env API (_reset_state, get_state_history) — will break on NeuroGym; must fix before Phase 2
- [Phase 3 planning]: Latent circuit rank selection for context-DM task needs verification against engellab/latentcircuit repo (research flag)
- [Phase 4 planning]: Nassar 2021 .mat file nested indexing not directly inspected — must run describe_mat_structure() before writing data loading code (research flag)

## Session Continuity

Last session: 2026-03-18T20:04:09Z
Stopped at: Completed 01-02-PLAN.md — pyproject.toml updated, PyMC/PyEM archived
Resume file: None
