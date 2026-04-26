---
phase: 03-latent-circuit-inference
plan: "06"
subsystem: analysis
tags: [pytorch, latentnet, circuit-inference, slurm, per-context, diagnostic]
gap_closure: true

# Dependency graph
requires: []
provides:
  - "scripts/analysis/fit_per_context_latent_circuits.py — per-context LatentNet fitting driver"
  - "cluster/run_per_context_fits.sh + cluster/run_per_context_one.slurm — SLURM submitter"
  - "output/circuit_analysis/per_context/context_{0,1}/ — per-context Q artifacts (PENDING AUTOPUSH)"
  - "output/circuit_analysis/per_context/per_context_results.json — conclusion enum (PENDING TASK 2)"
affects: ["03-08 (writeup re-commit — conclusion drives SC-2 narrative)"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-context slice pattern: load npz READ-ONLY, mask by labels['modality_context'], assert n_trials >= 100"
    - "Defensive W_rec load: assert W_hh.weight.shape == (64,64) and W_ih.weight.shape == (64,7) before extracting"
    - "cluster_same_seed_as_train eval: sigma_rec=0.15 default, no eval-mode override for comparability"

key-files:
  created:
    - "scripts/analysis/fit_per_context_latent_circuits.py"
    - "cluster/run_per_context_fits.sh"
    - "cluster/run_per_context_one.slurm"
    - "cluster/logs/per_context_jobs.txt (placeholder — populated by run_per_context_fits.sh on cluster)"
  modified:
    - "cluster/99_push_results.slurm (added per_context staging patterns)"

key-decisions:
  - "sigma_rec=0.15 default (no eval-mode override) — cluster_same_seed_as_train, matches Wave A pooled baseline for direct corr comparison"
  - "circuit_data.npz read READ-ONLY — slicing in-memory only, no writes, no race condition with 03-05"
  - "Defensive checkpoint load pattern: assert W_hh/W_ih shapes before extracting W_rec; halt+document if mismatch"
  - "afterany (not afterok) dependency for autopush — push fires even if one context fit fails"

patterns-established:
  - "Per-context diagnostic: slice by labels['modality_context'], fit independently, compare deltas vs pooled"
  - "Conclusion classification: STRUCTURAL_SEPARATION (both delta>=0.05), NO_SEPARATION (both |delta|<0.05), AMBIGUOUS (otherwise)"

# Metrics
duration: ~35min (local implementation + smoke test; cluster fits ~95min each, pending)
completed: 2026-04-26
---

# Phase 03 Plan 06: Per-Context Latent Circuit Fitting Summary

**PARTIAL SUMMARY — Task 1 (driver + cluster scripts + smoke test) complete; Task 2 (aggregation + conclusion + per_context_results.json) PENDING AUTOPUSH from cluster.**

Per-context LatentNet fitting driver implemented and smoke-tested; cluster scripts submitted (pending manual run on Monash M3); local smoke test passed (context=0, n_inits=1, epochs=5, corr=-0.089, nmse_y=7.23 — expected low quality at 5 epochs).

## Performance

- **Duration:** ~35 min (local tasks)
- **Started:** 2026-04-26T09:00:00Z (approx)
- **Completed (partial):** 2026-04-26T09:33:10Z
- **Tasks complete:** 1/2 (Task 2 awaiting cluster autopush)
- **Files created/modified:** 5

## Accomplishments (Task 1)

- Driver `fit_per_context_latent_circuits.py` implemented with argparse, defensive W_rec load (shape asserts), context slicing, ensemble + validation, JSON output (all Python builtins)
- `cluster/run_per_context_fits.sh` submitter: two parallel SLURM jobs (ctx 0 + ctx 1) + `afterany:JID0:JID1` autopush dependency
- `cluster/run_per_context_one.slurm` per-job script: GPU verification, env setup, python invocation
- `cluster/99_push_results.slurm` updated with per_context staging patterns
- Local smoke test passed: context 0, 500 trials sliced, W_rec (64,64) shape assertion passed, validation_results.json produced with all Python-typed values
- ruff check passes on driver script

## Task Commits

1. **Task 1A — Driver:** `4f77fc6` (feat(03-06): per-context latent circuit fitting driver)
2. **Task 1B+C — Cluster scripts (run_per_context_fits.sh + run_per_context_one.slurm):** `c5ebebb` (feat(03-06): cluster submitter for per-context fits)
3. **Task 1B+C — 99_push_results.slurm update:** `0670239` (feat(03-06): cluster submitter for per-context fits)
4. **Task 1E — JID placeholder:** `09e4ed9` (chore(03-06): record cluster JIDs placeholder)

**Task 2:** PENDING — to be executed by `/gsd:execute-phase 03 --gaps-only` after autopush delivers context_0 + context_1 validation_results.json.

## Files Created/Modified

- `scripts/analysis/fit_per_context_latent_circuits.py` — Per-context LatentNet fitting driver
- `cluster/run_per_context_fits.sh` — Two-job parallel SLURM submitter with autopush
- `cluster/run_per_context_one.slurm` — Per-context SLURM job script
- `cluster/99_push_results.slurm` — Added per_context staging patterns
- `cluster/logs/per_context_jobs.txt` — Placeholder JID log (populated on cluster)
- `output/circuit_analysis/per_context/smoke_test_ctx0/` — Local smoke test artifacts (not committed; gitignored per .npz rule; the .pt and .json files are gitignored under /output/* but the per_context/ dir is whitelisted — smoke_test subdir will not be staged by autopush)

## Smoke Test Results

```
Context 0: 500 trials selected
W_rec shape: (64, 64) — OK
n_inits=1, epochs=5, device=cpu
SUMMARY n_trials_used=500, corr=-0.0892, nmse_y=7.2304, status=SOFT-FAIL
```

(Low corr/high nmse expected at 5 epochs — this is a driver validation only, not a meaningful fit. Full cluster fit uses n_inits=100, epochs=500.)

## Decisions Made

- **sigma_rec=0.15, no eval-mode override:** Matches Wave A eval procedure exactly. Corr is reported on the same noise realization as training (cluster_same_seed_as_train). This ensures per-context corrs are directly comparable to pooled baseline 0.7833.
- **Circuit_data.npz READ-ONLY:** Sliced in-memory only. No writes. No race condition with 03-05's task_active_mask regeneration. The per-context fits use `u`, `z`, `y`, `labels['modality_context']` — all present in both pre-masked and post-masked versions.
- **afterany (not afterok) for autopush:** Push fires even if one context fit fails, allowing partial result recovery.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] matplotlib import order for ruff compliance**

- **Found during:** Task 1A (ruff check on driver script)
- **Issue:** `matplotlib.use("Agg")` placed between `from __future__ import annotations` and remaining imports caused ruff E402 (module level import not at top) and I001 (import sort) errors
- **Fix:** Moved `from __future__ import annotations` after module docstring (matching aggregate_n_latent_sweep.py pattern); added `# noqa: E402, I001` on post-Agg imports; removed unused `import sys`; fixed one E501 line-length violation
- **Files modified:** `scripts/analysis/fit_per_context_latent_circuits.py`
- **Verification:** `ruff check scripts/analysis/fit_per_context_latent_circuits.py` → "All checks passed!"
- **Committed in:** `4f77fc6`

---

**Total deviations:** 1 auto-fixed (Rule 3 - Blocking)
**Impact on plan:** Minor import ordering fix for ruff compliance. No logic changes.

## Cluster Submission Status

**PENDING — sbatch not available on local Windows machine.**

To submit from Monash M3:
```bash
bash cluster/run_per_context_fits.sh
```

This will:
1. Submit two parallel fits (context 0 + context 1, ~95 min each on GPU)
2. Wire afterany autopush via `cluster/99_push_results.slurm`
3. Overwrite `cluster/logs/per_context_jobs.txt` with actual JIDs

After autopush delivers `output/circuit_analysis/per_context/context_0/validation_results.json` and `context_1/validation_results.json`, run:
```
/gsd:execute-phase 03 --gaps-only
```
to trigger Task 2 (aggregation + conclusion + per_context_results.json + final SUMMARY).

## Task 2 Preview (PENDING)

When cluster results arrive, Task 2 will:
1. Read both `validation_results.json` files
2. Load Wave A baseline corr (0.7833) from `wave_a_selection.json`
3. Compute deltas and classify conclusion:
   - **STRUCTURAL_SEPARATION**: both delta >= 0.05 — RNN has two distinct low-rank circuits
   - **NO_SEPARATION**: both |delta| < 0.05 — pooled fit is appropriate abstraction
   - **AMBIGUOUS**: anything else (one lifted, one not)
4. Write `output/circuit_analysis/per_context/per_context_results.json`
5. Update this SUMMARY with the full per-context table and conclusion

**Conclusion drives 03-08 story commitment:** STRUCTURAL_SEPARATION → SC-2 narrative shifts to "pooled fit was wrong abstraction" (Story 1 reframe); NO_SEPARATION/AMBIGUOUS → per-context hypothesis ruled out (Story 2 or diagnostic note).

## Issues Encountered

None beyond the ruff import ordering (auto-fixed above).

## Next Phase Readiness

- Task 1 complete, committed, smoke tested
- Cluster submission pending (manual run of `bash cluster/run_per_context_fits.sh` on Monash M3)
- Task 2 blocked on autopush delivery — will be picked up by next `/gsd:execute-phase 03 --gaps-only` run
- 03-08 story commitment blocked on Task 2 conclusion classification

---
*Phase: 03-latent-circuit-inference*
*Completed (partial): 2026-04-26*
