---
phase: 04-bayesian-model-fitting
plan: "02"
subsystem: bayesian
tags: [numpyro, jax, arviz, mcmc, diagnostics, parameter-recovery, pearson-r, nuts]

# Dependency graph
requires:
  - phase: 04-01
    provides: "reduced_bayesian_model, run_mcmc, prior_sampler, simulate_synthetic_data, 4-CPU XLA setup"
provides:
  - "diagnostics.py: run_diagnostics (R-hat/ESS/divergences via ArviZ), fit_with_retry (retry-once loop), make_fit_summary (10KB JSON shape), to_jsonable helper"
  - "09a_param_recovery.py: end-to-end recovery driver with --smoke flag; per-condition fits (CP+OB averaged)"
  - "Smoke recovery completed: 4 datasets x 2 conditions = 8 MCMC fits; recovery_report.json + 5 scatter PNGs on disk"
  - "Full 50-dataset run queued for overnight (background_task_id=bcaxsbh0c, started 2026-04-30T07:10:31Z)"
  - "REQUIREMENTS.md BAYES-03 updated: ESS_bulk wording + divergence-documented policy"
  - "ROADMAP.md SC-3/SC-4 updated: divergences documented, not gated"
affects:
  - "04-03: imports fit_with_retry, make_fit_summary from nn4psych.bayesian; Task 2 GATES on full-run recovery_report.json (all r >= 0.85)"
  - "04-04b: imports same diagnostics surface for RNN cohort fits"

# Tech tracking
tech-stack:
  added: []  # arviz already pinned in 04-01; scipy already in env; no new packages
  patterns:
    - "M3 fix (ArviZ 0.23.4): az.summary uses stat_focus='stats'/'diagnostics', NOT kind= kwarg"
    - "stat_focus first, fallback to single-call column-slice if stat_focus not accepted (future-proofing)"
    - "to_jsonable() recursive numpy scalar converter applied before json.dump (recurring 02-03 lesson)"
    - "Per-condition recovery: CP+OB fits per dataset, posterior means averaged — validates actual human fitting pipeline"
    - "get_extra_fields() always returns {'diverging': ...} in this NumPyro version (no explicit extra_fields kwarg needed for divergence access)"

key-files:
  created:
    - src/nn4psych/bayesian/diagnostics.py
    - tests/test_diagnostics.py
    - scripts/data_pipeline/09a_param_recovery.py
    - data/processed/bayesian/param_recovery_smoke/recovery_report.json
    - data/processed/bayesian/param_recovery_smoke/figures/recovery_H.png
    - data/processed/bayesian/param_recovery_smoke/figures/recovery_LW.png
    - data/processed/bayesian/param_recovery_smoke/figures/recovery_UU.png
    - data/processed/bayesian/param_recovery_smoke/figures/recovery_sigma_motor.png
    - data/processed/bayesian/param_recovery_smoke/figures/recovery_sigma_LR.png
  modified:
    - src/nn4psych/bayesian/__init__.py
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md

key-decisions:
  - "stat_focus kwarg path used (ArviZ 0.23.4 confirmed), not fallback single-call — M3 fix validated at runtime"
  - "NumPyro get_extra_fields() always returns diverging field in this version; warning test used monkeypatch to simulate absence"
  - "Per-condition recovery design: one MCMC per (dataset, condition), posterior means averaged — matches human fitting pipeline"
  - "Smoke N=4 r values are NOT formal BAYES-06 evidence (N too small for reliable Pearson r); full 50-dataset run is the gate"
  - "Smoke MCMC config (200 warmup/samples, 2 chains) caused all 8 fits to fail convergence gates — expected at smoke settings"

patterns-established:
  - "nn4psych.bayesian canonical diagnostics: fit_with_retry returns (mcmc, status, attempts); make_fit_summary is Phase 5 ingestion contract"
  - "Recovery driver pattern: prior_sampler → simulate_synthetic_data (CP+OB) → fit_with_retry x2 → average posterior means → pearsonr"

# Metrics
duration: 51min (plus 33min smoke run + full run ongoing)
completed: 2026-04-30
---

# Phase 4 Plan 02: Diagnostics + Parameter Recovery Summary

**ArviZ-backed MCMC diagnostics module (run_diagnostics, fit_with_retry, make_fit_summary) with stat_focus= M3 fix; smoke recovery on 4 synthetic datasets completed; full 50-dataset run queued for overnight**

## Performance

- **Duration:** 51 min (code + smoke run: 33 min; full run ongoing)
- **Started:** 2026-04-30T06:19:39Z
- **Completed:** 2026-04-30T07:11:03Z (smoke complete; full run queued)
- **Tasks:** 3 / 3
- **Files created/modified:** 12

## Accomplishments

- Created `src/nn4psych/bayesian/diagnostics.py` with three public functions (`run_diagnostics`, `fit_with_retry`, `make_fit_summary`) and `to_jsonable` helper — reusable across 04-03/04-04b
- M3 fix implemented and verified: `az.summary` called with `stat_focus='stats'/'diagnostics'` (ArviZ 0.23.4 pattern); fallback branch included for version resilience
- 5 unit tests in `tests/test_diagnostics.py` — all passing, including monkeypatched retry-fallback and missing-diverging-field warning path
- Smoke parameter recovery completed: 4 datasets x 2 conditions = 8 MCMC fits; `recovery_report.json` + 5 scatter PNGs on disk
- Full 50-dataset overnight run queued (background_task_id=bcaxsbh0c, started 2026-04-30T07:10:31Z)
- REQUIREMENTS.md and ROADMAP.md SC-3/SC-4 updated to reflect "document divergences, don't gate" decision

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement diagnostics module** - `4eb0abf` (feat)
2. **Task 2: Parameter recovery driver and smoke run** - `a0a3274` (feat)
3. **Task 3: Update REQUIREMENTS/ROADMAP SC wording** - `6cf2209` (chore)

## Files Created/Modified

- `src/nn4psych/bayesian/diagnostics.py` — NEW: run_diagnostics, fit_with_retry, make_fit_summary, to_jsonable
- `src/nn4psych/bayesian/__init__.py` — UPDATED: 4 new exports added to imports + __all__
- `tests/test_diagnostics.py` — NEW: 5 unit tests (all passing)
- `scripts/data_pipeline/09a_param_recovery.py` — NEW: recovery driver with --smoke flag, per-condition fits, scatter plots
- `data/processed/bayesian/param_recovery_smoke/recovery_report.json` — NEW: smoke recovery (N=4) aggregate
- `data/processed/bayesian/param_recovery_smoke/figures/*.png` — NEW: 5 scatter PNGs
- `data/processed/bayesian/param_recovery_smoke/per_fit/*.json` — NEW: 8 per-fit JSONs
- `.planning/REQUIREMENTS.md` — UPDATED: BAYES-03 ESS_bulk + divergence-documented wording
- `.planning/ROADMAP.md` — UPDATED: SC-3/SC-4 + Phase 4 Note (2026-04-29)

## Smoke Recovery Results (N=4 datasets)

**Config:** 2 chains x 200 warmup x 200 samples x 100 trials/condition

| Parameter    | r      | Pass (r >= 0.85) |
|-------------|--------|-----------------|
| H           | 0.900  | PASS            |
| LW          | -0.469 | FAIL            |
| UU          | 0.283  | FAIL            |
| sigma_motor | 0.772  | FAIL            |
| sigma_LR    | 0.115  | FAIL            |

**Overall:** 1/5 parameters pass gate
**Failed fits (convergence gates):** 8/8 — ALL smoke fits failed R-hat/ESS gates
**Interpretation:** Expected at N=4 with smoke MCMC settings (200 warmup/samples). Pearson r is unreliable at N=4 even for H's apparent pass. Formal BAYES-06 evidence requires full 50-dataset run.

**Note on smoke convergence failures:** All 8 fits (dataset x condition) triggered the retry mechanism but still failed both R-hat and ESS gates after retry (400 warmup, 200 samples at 0.99 target_accept). This is expected at smoke settings — the posterior geometry of the Nassar model requires substantially more warmup to mix properly. The full run uses 2000/4000 warmup with 2000 samples which should achieve convergence.

## Full Overnight Run Status

**Background task ID:** bcaxsbh0c
**Started:** 2026-04-30T07:10:31Z
**Command:** `python scripts/data_pipeline/09a_param_recovery.py --output_dir data/processed/bayesian/param_recovery`
**Config:** 50 datasets, 4 chains x 2000 warmup x 2000 samples, 100 trials/condition
**Expected wall time:** ~17h worst case (100 MCMC fits sequentially at ~10 min each with retry)
**Output will appear at:** `data/processed/bayesian/param_recovery/recovery_report.json`

**BAYES-06 IS NOT CLOSED:** Full closure requires the 50-dataset run completing with per-parameter r >= 0.85. This gates 04-03 Task 2 (prior-change re-validation branch). Per B2/M4 scope clarification: this plan is DONE; BAYES-06 closure is a separate milestone.

## ArviZ kwarg path used (M3 fix)

The `stat_focus` path was used successfully (not the fallback). ArviZ 0.23.4 accepts `stat_focus='stats'` and `stat_focus='diagnostics'` in `az.summary`. The `kind=` kwarg raises `TypeError` in this version — would have been caught during `make_fit_summary` before `json.dumps`.

Fallback (single `az.summary` call + column-slice) is in the code but was NOT exercised — future-proofing for ArviZ API changes.

## Decisions Made

1. **stat_focus kwarg path validated:** ArviZ 0.23.4 accepts `stat_focus` — the M3 fix is the correct implementation path. Fallback included in code for version resilience.

2. **Smoke convergence failure expected:** All 8 smoke fits failed R-hat/ESS gates with 200 warmup/samples (smoke config). This is a known limitation of the reduced Bayesian model's posterior geometry at minimal MCMC settings. Full run (2000 warmup) will give proper convergence.

3. **NumPyro extra_fields discovery:** `mcmc.get_extra_fields()` always returns `{'diverging': ...}` in this NumPyro version, even without explicit `extra_fields=('diverging',)` kwarg. The warning code path is preserved for future-proofing; test uses monkeypatch to simulate absence.

4. **Per-condition design validated:** Fitting CP and OB conditions separately and averaging posterior means mirrors the actual human fitting pipeline (04-03 will do the same per-subject x per-condition). This is the correct recovery validation design per CONTEXT.md.

5. **Smoke r values not formal evidence:** N=4 is too small for reliable Pearson r. H r=0.900 at N=4 has a very wide confidence interval. The 50-dataset run is the formal gate for BAYES-06.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_run_diagnostics_warns_without_extra_fields for NumPyro behavior**
- **Found during:** Task 1 (test_diagnostics.py verification)
- **Issue:** Test assumed running MCMC without `extra_fields=('diverging',)` would make `get_extra_fields()` return a dict without 'diverging'. In practice, NumPyro always includes 'diverging' in `get_extra_fields()`. Additionally, `az.from_numpyro()` internally calls `get_extra_fields(group_by_chain=True)`, so a naive `lambda: None` patch broke ArviZ.
- **Fix:** Updated test to use monkeypatch that strips 'diverging' from the no-kwarg call path only (while passing through the `group_by_chain=True` call for ArviZ compatibility).
- **Files modified:** tests/test_diagnostics.py
- **Verification:** Test passes (1/1)
- **Committed in:** 4eb0abf (Task 1 commit)

**2. [Rule 3 - Blocking] Fixed ModuleNotFoundError for nn4psych.bayesian in script context**
- **Found during:** Task 2 (smoke run attempt)
- **Issue:** `scripts/data_pipeline/09a_param_recovery.py` failed with `ModuleNotFoundError: No module named 'envs'` because `nn4psych/__init__.py` does `from envs import PIE_CP_OB_v2`, and `envs/` module is only findable when the project root is on sys.path.
- **Fix:** Added both `src/` and project root to sys.path at script startup. This matches the pattern needed for pytest (which adds both) and ensures `envs` is importable alongside `nn4psych.bayesian`.
- **Files modified:** scripts/data_pipeline/09a_param_recovery.py
- **Verification:** Smoke run completed successfully end-to-end
- **Committed in:** a0a3274 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes necessary for correct test behavior and script execution. No scope creep.

## Issues Encountered

- Smoke run took 33 minutes (vs estimated 5-15 min) due to retry mechanism triggering on all 8 fits — MCMC with 200 warmup consistently fails R-hat/ESS gates, triggering retry at 400 warmup. This doubles the smoke wall time. Not a bug — expected behavior. The full run timing estimate (~17h) is based on 2000/4000 warmup which should converge without retry; actual may be faster.

## Next Phase Readiness

**Ready for 04-03:**
- `nn4psych.bayesian.fit_with_retry` and `make_fit_summary` are importable and tested
- Per-fit JSON shape established: per-parameter mean/median/sd/hdi_2.5/97.5/rhat/ess_bulk + status + attempts + n_divergences
- REQUIREMENTS.md BAYES-03 wording aligned with divergence-documenting policy

**Blockers for 04-03:**
- 04-03 Task 1: Brain2021Code download (human .mat files not on disk) — this is the first gating step in 04-03
- 04-03 Task 2 (prior-change re-validation): GATES on `data/processed/bayesian/param_recovery/recovery_report.json` being present with all per-parameter r >= 0.85. This requires the overnight run to complete.

**Open items:**
- Full overnight run (background_task_id=bcaxsbh0c): if r < 0.85 for any parameter, first diagnostic is tau update equation form (RESEARCH.md Pitfall 1 — compute_rbo_forward Eq. 5 variant). Do NOT start 04-03 Task 2 until full run completes.

---
*Phase: 04-bayesian-model-fitting*
*Completed: 2026-04-30*
