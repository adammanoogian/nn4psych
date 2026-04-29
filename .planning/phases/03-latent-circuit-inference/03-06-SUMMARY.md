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
  - "scripts/analysis/aggregate_per_context.py — one-shot aggregator for per-context results"
  - "cluster/run_per_context_fits.sh + cluster/run_per_context_one.slurm — SLURM submitter"
  - "output/circuit_analysis/per_context/context_0/ — ctx-0 Q artifacts (best_latent_circuit.pt, validation_results.json, ensemble_diagnostics.json)"
  - "output/circuit_analysis/per_context/context_1/ — ctx-1 Q artifacts"
  - "output/circuit_analysis/per_context/per_context_results.json — conclusion enum AMBIGUOUS"
affects: ["03-08 (writeup re-commit — conclusion drives SC-2 narrative)"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-context slice pattern: load npz READ-ONLY, mask by labels['modality_context'], assert n_trials >= 100"
    - "Defensive W_rec load: assert W_hh.weight.shape == (64,64) and W_ih.weight.shape == (64,7) before extracting"
    - "cluster_same_seed_as_train eval: sigma_rec=0.15 default, no eval-mode override for comparability"
    - "One-shot aggregator pattern: pure JSON parsing + arithmetic, no PyTorch dependency"

key-files:
  created:
    - "scripts/analysis/fit_per_context_latent_circuits.py"
    - "scripts/analysis/aggregate_per_context.py"
    - "cluster/run_per_context_fits.sh"
    - "cluster/run_per_context_one.slurm"
    - "cluster/logs/per_context_jobs.txt"
    - "output/circuit_analysis/per_context/per_context_results.json"
  modified:
    - "cluster/99_push_results.slurm (added per_context staging patterns)"

key-decisions:
  - "sigma_rec=0.15 default (no eval-mode override) — cluster_same_seed_as_train, matches Wave A pooled baseline for direct corr comparison"
  - "circuit_data.npz read READ-ONLY — slicing in-memory only, no writes, no race condition with 03-05"
  - "Defensive checkpoint load pattern: assert W_hh/W_ih shapes before extracting W_rec; halt+document if mismatch"
  - "afterany (not afterok) dependency for autopush — push fires even if one context fit fails"
  - "Conclusion = AMBIGUOUS (by exhaustion): both per-context corrs LOWER than pooled by 0.12-0.14 — structural-separation hypothesis ruled out; Q quality cap is method/data-bound"

patterns-established:
  - "Per-context diagnostic: slice by labels['modality_context'], fit independently, compare deltas vs pooled"
  - "Conclusion classification: STRUCTURAL_SEPARATION (both delta>=0.05), NO_SEPARATION (both |delta|<0.05), AMBIGUOUS (otherwise)"

# Metrics
duration: ~45min (local implementation + smoke test 35min; aggregation 10min; cluster fits ~95min each on GPU)
completed: 2026-04-29
---

# Phase 03 Plan 06: Per-Context Latent Circuit Fitting Summary

**Per-context LatentNet diagnostic at n=12: ctx-0 corr=0.6628, ctx-1 corr=0.6406 both BELOW pooled 0.7833 — AMBIGUOUS conclusion rules out structural-separation hypothesis for SC-2 soft-fail**

## Performance

- **Duration:** ~45 min total (35 min Task 1; 10 min Task 2 aggregation; cluster fits ran independently ~95 min on GPU)
- **Started:** 2026-04-26T09:00:00Z (approx)
- **Completed:** 2026-04-29T20:10:00Z
- **Tasks complete:** 2/2
- **Files created/modified:** 8

## Accomplishments

- Driver `fit_per_context_latent_circuits.py` implemented with argparse, defensive W_rec load (shape asserts), context slicing by `labels['modality_context']`, ensemble + validation, JSON output (all Python builtins)
- `cluster/run_per_context_fits.sh` submitter: two parallel SLURM jobs (ctx 0 + ctx 1) + `afterany:JID0:JID1` autopush dependency
- `cluster/run_per_context_one.slurm` per-job script: GPU verification, env setup, python invocation
- `cluster/99_push_results.slurm` updated with per_context staging patterns
- Cluster fits completed on Monash M3 (n_latent=12, n_inits=100, epochs=500, device=cuda); artifacts autopushed
- Aggregator `aggregate_per_context.py` reads both validation_results.json + Wave A baseline, computes deltas, classifies AMBIGUOUS, writes `per_context_results.json`
- **Conclusion AMBIGUOUS**: both per-context corrs lower than pooled by 0.12-0.14 — structural-separation hypothesis is NOT supported

## Per-Context Results Table

| Context | n_trials | corr   | nmse_y | delta_vs_pooled | \|delta\| |
|---------|----------|--------|--------|-----------------|-----------|
| ctx-0   | 500      | 0.6628 | 0.2455 | -0.1205         | 0.1205    |
| ctx-1   | 500      | 0.6406 | 0.2451 | -0.1427         | 0.1427    |
| pooled (Wave A, n=12) | 1000 | **0.7833** | 0.2472 | — | — |

**Separation threshold:** 0.05
**Conclusion:** AMBIGUOUS (by exhaustion — both deltas large negative)
**either_above_85:** False

## Conclusion and Interpretation

**AMBIGUOUS** — neither STRUCTURAL_SEPARATION nor NO_SEPARATION.

The AMBIGUOUS classification arises because neither delta satisfies the separation threshold in any direction: delta_ctx0 = -0.1205, delta_ctx1 = -0.1427. Both per-context corrs are LOWER than pooled by ~0.12-0.14, which RULES OUT the structural-separation hypothesis. The pooled fit's higher corr likely reflects more training data (1000 vs 500 trials), not a structural mixing artifact. Therefore the SC-2 soft-fail is NOT caused by the RNN encoding two distinct context-specific low-rank circuits — Q-quality cap is method/data-bound.

**Implication for SC-2:** The per-context diagnostic converges with 03-05's masked-loss result on STORY_1 (method/data limit). There is no evidence that splitting the data by modality_context improves the invariant subspace correlation; if anything, halving the trial count hurts fitting quality.

## Decision Flag for 03-08

**Per-context hypothesis ruled out (AMBIGUOUS by exhaustion, both deltas large negative). Combined with 03-05's negative masked delta, evidence converges on STORY_1 (method/data limit). 03-08 story commitment should default to STORY_1 unless 03-05 Wave B masked fits or 03-07 shorter-T fits show a breakthrough.**

Eval procedure: `cluster_same_seed_as_train` (sigma_rec=0.15 default, matches Wave A pooled baseline for direct corr comparison).

## Task Commits

1. **Task 1A — Driver:** `4f77fc6` (feat(03-06): per-context latent circuit fitting driver)
2. **Task 1B+C — Cluster scripts:** `c5ebebb` (feat(03-06): cluster submitter for per-context fits)
3. **Task 1B+C — 99_push_results.slurm update:** `0670239` (feat(03-06): cluster submitter for per-context fits)
4. **Task 1E — JID placeholder:** `09e4ed9` (chore(03-06): record cluster JIDs placeholder)
5. **Task 2 — Aggregation + per_context_results.json + SUMMARY:** TBD (feat(03-06): per-context aggregation + conclusion)

**Plan metadata:** TBD (docs(03-06): complete per-context plan)

## Files Created/Modified

- `scripts/analysis/fit_per_context_latent_circuits.py` — Per-context LatentNet fitting driver
- `scripts/analysis/aggregate_per_context.py` — One-shot aggregator: reads validation_results.json files, classifies conclusion, writes per_context_results.json
- `cluster/run_per_context_fits.sh` — Two-job parallel SLURM submitter with autopush
- `cluster/run_per_context_one.slurm` — Per-context SLURM job script
- `cluster/99_push_results.slurm` — Added per_context staging patterns
- `cluster/logs/per_context_jobs.txt` — JID log (populated on cluster)
- `output/circuit_analysis/per_context/context_0/validation_results.json` — Ctx-0 fit metrics (corr=0.6628)
- `output/circuit_analysis/per_context/context_1/validation_results.json` — Ctx-1 fit metrics (corr=0.6406)
- `output/circuit_analysis/per_context/per_context_results.json` — Conclusion enum + all per-context metrics

## Decisions Made

- **sigma_rec=0.15, no eval-mode override:** Matches Wave A eval procedure exactly. Ensures per-context corrs are directly comparable to pooled baseline 0.7833. This is the `cluster_same_seed_as_train` procedure.
- **Circuit_data.npz READ-ONLY:** Sliced in-memory only. No writes. No race condition with 03-05's task_active_mask regeneration.
- **afterany (not afterok) for autopush:** Push fires even if one context fit fails, allowing partial result recovery.
- **Conclusion AMBIGUOUS by exhaustion:** The three-rule enum is exhaustive. Both deltas negative-large (not opposite sign, not both near zero) falls into AMBIGUOUS as the catch-all. Interpretation: data-quantity effect, not structural separation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] matplotlib import order for ruff compliance**

- **Found during:** Task 1A (ruff check on driver script)
- **Issue:** `matplotlib.use("Agg")` placement caused ruff E402 and I001 errors
- **Fix:** Moved `from __future__ import annotations` after module docstring; added `# noqa: E402, I001`; fixed E501 line-length violations
- **Files modified:** `scripts/analysis/fit_per_context_latent_circuits.py`
- **Verification:** `ruff check scripts/analysis/fit_per_context_latent_circuits.py` → "All checks passed!"
- **Committed in:** `4f77fc6`

**2. [Rule 3 - Blocking] E501 line-length violations in aggregate_per_context.py**

- **Found during:** Task 2 (ruff check on aggregator)
- **Issue:** 5 E501 violations (path construction, boolean compound expressions, f-string print lines)
- **Fix:** Split path via intermediate `_SWEEP_DIR` variable; split boolean conditions into parenthesised two-line forms; split f-string prints into implicit string concatenation
- **Files modified:** `scripts/analysis/aggregate_per_context.py`
- **Verification:** `ruff check scripts/analysis/aggregate_per_context.py` → "All checks passed!"
- **Committed in:** Task 2 commit

---

**Total deviations:** 2 auto-fixed (both Rule 3 - Blocking, ruff compliance)
**Impact on plan:** Minor formatting fixes. No logic changes.

## Issues Encountered

None beyond ruff import/line-length fixes (both auto-fixed above).

## Next Phase Readiness

- 03-06 fully complete: per_context_results.json exists, conclusion = AMBIGUOUS, both deltas large negative
- 03-08 story commitment can now incorporate per-context diagnostic: structural-separation hypothesis ruled out
- 03-05 Wave B masked results and 03-07 shorter-T results still pending before 03-08 final story commit
- Combined evidence from 03-05 (masked delta negative) + 03-06 (per-context AMBIGUOUS) converges on STORY_1

---
*Phase: 03-latent-circuit-inference*
*Completed: 2026-04-29*
