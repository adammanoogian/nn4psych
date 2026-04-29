---
phase: 03-latent-circuit-inference
plan: "05"
subsystem: analysis
gap_closure: true
tags: [latent-circuit, masked-loss, neurogym, slurm, pytorch, numpy]

# Dependency graph
requires:
  - phase: 03-latent-circuit-inference
    provides: "Wave A n_latent sweep (03-03), circuit_data.npz (T=75, n=1000), model_context_dm_dual.pth"
provides:
  - "task_active_mask in circuit_data.npz: (1000, 75) bool, True during stimulus+decision"
  - "fit_latent_circuit_ensemble task_active_mask kwarg: masked NMSE_y + nmse_y_full cross-comparability"
  - "08_infer_latent_circuits.py --masked flag with n_latent_sweep_masked/ isolation"
  - "cluster/run_n_latent_sweep_masked.sh: 3-rank sweep submitter with MASKED=1 autopush"
  - "cluster/run_circuit_ensemble.sh: MASKED=1 assertion + masked-loss integration"
  - "scripts/analysis/aggregate_n_latent_sweep_masked.py: masked Pareto aggregator"
  - "pareto_curve_masked.json: 3-rank sweep results, masked:true, comparable_full_pareto ref"
  - "wave_a_masked_selection.json: chosen_rank=12, corr=0.5699, crossed_85=false, delta=-0.2134"
affects:
  - 03-07 (shorter-T regen — conditional skip if crossed_85=true; runs since false, but negative delta is strong prior it won't help)
  - 03-08 (writeup re-commit — reads wave_a_masked_selection.json; story tilts toward STORY_1 method/data limit)
  - 03-06 (per-context diagnostic — independent, parallel track)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Masked NMSE loss: broadcast (n_trials,T,1) float mask over residuals, divide by masked denom"
    - "per-step in_period() query BEFORE env.step() to capture current timestep's period"
    - "try/except around in_period() with all-True fallback + RuntimeWarning"
    - "best_nmse_y_full always computed alongside masked nmse_y for Wave A cross-comparability"
    - "gitignore un-ignore of circuit_data.npz for cluster pull (*.npz rule overridden)"
    - "pre_masked_backup at fixed name (not timestamped) to prevent git pollution"
    - "MASKED=1 assertion in cluster script: KeyError with pull instructions before silent fallback"

key-files:
  created:
    - scripts/analysis/aggregate_n_latent_sweep_masked.py
    - cluster/run_n_latent_sweep_masked.sh
    - cluster/logs/n_latent_sweep_masked_jobs.txt
    - output/circuit_analysis/n_latent_sweep_masked/pareto_curve_masked.json
    - output/circuit_analysis/n_latent_sweep_masked/pareto_curve_masked.png
    - output/circuit_analysis/n_latent_sweep_masked/wave_a_masked_selection.json
    - .planning/phases/03-latent-circuit-inference/03-05-SUMMARY.md
  modified:
    - src/nn4psych/analysis/circuit_inference.py
    - scripts/data_pipeline/08_infer_latent_circuits.py
    - cluster/run_circuit_ensemble.sh
    - data/processed/rnn_behav/circuit_data.npz
    - .gitignore

key-decisions:
  - "Fixation excluded from task_active_mask (sensorimotor-only, low info for circuit fitting)"
  - "Delay period excluded (blank maintenance, not discrimination signal)"
  - "task_active_mask=None in fit_latent_circuit_ensemble preserves bit-identical Wave A behavior"
  - "Mini-batch masked loop (batch_size=128, mirrors LatentNet.fit()) — not full-batch"
  - "circuit_data.npz un-ignored via explicit !/data/.../circuit_data.npz rule (cluster needs pull)"
  - "circuit_data.npz.pre_masked_backup at deterministic name — gitignored, one fixed entry"
  - "Masked corr WORSE than Wave A baseline (delta=-0.2134) — padding hypothesis ruled out"
  - "Story direction: STORY_1 (method/data limit) — 03-08 will commit this narrative"
  - "03-07 still runs (crossed_85=false) but negative delta is strong prior shorter T won't help"

patterns-established:
  - "Masked loss: zero residuals at non-active timesteps, divide by sum of masked denom"
  - "nmse_y_full always reported alongside masked nmse_y for cross-wave comparability"
  - "MASKED=1 assertion in cluster script: KeyError with pull instructions before silent fallback"
  - "Negative delta (masked < full) rules out hypothesis even if spread > 0.05"

# Metrics
duration: "~8 hours local execution (Tasks 1-2) + 93–191 min cluster (Task 3 sweep) + aggregation"
completed: 2026-04-29
---

# Phase 03 Plan 05: Masked-Loss Fitting (Gap 1) Summary

**Masked-loss LatentNet sweep (n={8,12,16}, 100 inits, 500 epochs, MASKED=1) completed on cluster; chosen rank n=12 corr=0.5699 is WORSE than Wave A baseline (0.7833, delta=-0.2134) — padding hypothesis ruled out, story tilts to STORY_1 method/data limit**

## Performance

- **Duration:** ~8 hours local (Tasks 1-2, 2026-04-26) + cluster GPU time per rank (93–191 min); Task 3 aggregation ~2 min (2026-04-29)
- **Started:** 2026-04-26
- **Completed:** 2026-04-29 (Task 3 after cluster autopush landed)
- **Tasks completed:** 3/3
- **Files created:** 7 (aggregator, cluster scripts, JSON artifacts, PNG, SUMMARY)
- **Files modified:** 5 (circuit_inference.py, 08_infer_latent_circuits.py, run_circuit_ensemble.sh, circuit_data.npz, .gitignore)

## Accomplishments

- Augmented `collect_circuit_data` with per-step `in_period()` mask query (stimulus + decision only; fixation and delay excluded). Backward-compat fallback to all-True with RuntimeWarning if env lacks period timing API.
- Added `task_active_mask` kwarg to `fit_latent_circuit_ensemble` (default None = bit-identical Wave A behavior). Masked variant uses mini-batch loop (batch_size=128) matching LatentNet.fit() exactly, with masked NMSE_y and mse_z. Always reports `nmse_y_full` alongside masked metric for cross-wave comparability.
- Regenerated `circuit_data.npz` from canonical model: shape (1000,75), mean_active=32.5 steps/trial (min=20, max=43). Un-ignored in .gitignore so cluster can `git pull` it.
- Added `--masked` CLI flag to `08_infer_latent_circuits.py`. Masked path writes to `n_latent_sweep_masked/` (no collision with Wave A's `n_latent_sweep/`). `"masked": true` in all JSON artifacts.
- Created `cluster/run_n_latent_sweep_masked.sh`: 3-rank sweep (n=8,12,16) with MASKED=1, job name prefix `circuit_masked_n{N}`, autopush via `99_push_results.slurm`.
- Added MASKED=1 assertion to `cluster/run_circuit_ensemble.sh`: if cluster's npz lacks `task_active_mask` key, raises KeyError with pull instructions. Prevents silent fallback to full-T fitting on stale checkout.
- Created `scripts/analysis/aggregate_n_latent_sweep_masked.py` (mirrors `aggregate_n_latent_sweep.py`; same argmax corr selection rule). Produces `pareto_curve_masked.json` and `wave_a_masked_selection.json`.
- **Key scientific result:** Masked-loss corr at chosen rank n=12 is 0.5699, compared to Wave A baseline 0.7833 (delta=-0.2134). Masking task-inactive timesteps *hurt* connectivity alignment — the padding-noise hypothesis for the SC-2 soft-fail is ruled out. Story tilts toward STORY_1 (method/data limit).

## Task Commits

1. **Task 1 (.gitignore)** — `b234afb` (chore: gitignore pre_masked_backup)
2. **Task 1 (circuit_inference.py)** — `573caed` (feat: task_active_mask + masked-loss fit)
3. **Task 1 (circuit_data.npz)** — `715b24f` (data: add task_active_mask to circuit_data.npz)
4. **Task 2 (CLI + cluster)** — `480cb74` (feat: --masked CLI + cluster sweep submitter)
5. **Task 2 (JIDs placeholder)** — `108d321` (chore: record cluster JIDs placeholder)
6. **Task 3 (aggregator + artifacts)** — `f4574c8` (feat(03-05): aggregator + wave_a_masked_selection.json)
7. **Plan metadata** — pending (this commit)

## Per-Rank Pareto Table (Masked Sweep)

| n_latent | masked_corr | nmse_y_full | masked_nmse_y | trial_avg_r2 | wall_min |
|----------|-------------|-------------|---------------|--------------|----------|
| 8        | 0.4296      | 0.4627      | 0.0458        | 0.9667       | 93.5     |
| **12**   | **0.5699**  | **0.3909**  | **0.0352**    | **0.9754**   | **190.6**|
| 16       | 0.4558      | 0.4036      | 0.0355        | 0.9696       | 157.8    |

**Chosen rank:** n=12 (argmax invariant_subspace_corr, same rule as Wave A)

### Wave A Comparison (n=12)

| Metric                   | Wave A (full T) | Masked sweep | Delta     |
|--------------------------|-----------------|--------------|-----------|
| invariant_subspace_corr  | 0.7833          | 0.5699       | **-0.2134** |
| best_nmse_y              | 0.2472          | 0.0352 (masked) / 0.3909 (full) | — |
| trial_avg_r2_full        | 0.9764          | 0.9754       | -0.001    |
| n_inits / epochs         | 100 / 500       | 100 / 500    | —         |
| eval_procedure           | cluster_same_seed | cluster_same_seed | — |

**crossed_85_threshold:** false (0.5699 < 0.85)

**Pareto spread:** 0.1403 (corr varies across ranks — not flat — but negative delta is decisive)

### Decision for 03-07 (Shorter-T Regen)

Masked corr WORSE than Wave A baseline (delta=-0.2134). Padding hypothesis ruled out — focusing loss on task-active timesteps did not improve connectivity alignment. **Tilt Phase 3.1 story toward STORY_1 (method/data limit).** 03-07 (shorter T regen) will still RUN per its conditional skip rule (crossed_85=false), but the negative delta is a strong prior that shorter T also won't help.

## Files Created/Modified

- `scripts/analysis/aggregate_n_latent_sweep_masked.py` — masked Pareto aggregator (new)
- `output/circuit_analysis/n_latent_sweep_masked/pareto_curve_masked.json` — 3-rank Pareto (new)
- `output/circuit_analysis/n_latent_sweep_masked/pareto_curve_masked.png` — two-curve overlay (new)
- `output/circuit_analysis/n_latent_sweep_masked/wave_a_masked_selection.json` — chosen rank + threshold flag (new)
- `src/nn4psych/analysis/circuit_inference.py` — task_active_mask collection + masked-loss fit variant
- `data/processed/rnn_behav/circuit_data.npz` — augmented with task_active_mask (1000,75) bool
- `scripts/data_pipeline/08_infer_latent_circuits.py` — --masked flag, masked path, JSON field
- `cluster/run_n_latent_sweep_masked.sh` — new 3-rank masked sweep submitter
- `cluster/run_circuit_ensemble.sh` — MASKED config + assertion + masked fit call
- `.gitignore` — un-ignore circuit_data.npz; gitignore .pre_masked_backup

## Decisions Made

- **Fixation excluded from mask:** fixation is sensorimotor-only (hold position), carries no discrimination signal. Delay also excluded (blank working-memory maintenance, no new input). Only stimulus+decision are task-active.
- **task_active_mask=None preserves Wave A behavior:** verified by smoke test — mask=None produces identical nmse_y to the unmasked path.
- **Mini-batch masked loop:** uses batch_size=128, randomized per epoch, with connectivity_masks() after each batch — mirrors LatentNet.fit() exactly to preserve training dynamics. Not full-batch (which would differ in Adam gradient statistics).
- **circuit_data.npz committed to git:** unusual for .npz but necessary so the cluster can git pull the mask-augmented file without triggering the flock guard's full retrain (which would produce a file without task_active_mask).
- **Fixed backup name:** `circuit_data.npz.pre_masked_backup` (not timestamped). Re-runs overwrite the backup; the legacy_raw_logits_600 backup covers deeper history.
- **eval_procedure=cluster_same_seed_as_train:** masked sweep uses same evaluation procedure as Wave A for valid cross-comparison (both compute corr immediately after training on same noise realization).
- **Negative delta → padding hypothesis ruled out:** even though Pareto spread (0.14) is above the "flat" threshold (0.05), the *direction* of the delta (-0.21) is the decisive signal. Masking *hurt* corr. The story direction for 03-08 is STORY_1 (method/data limit), not STORY_0 (SC-2 cleared).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] circuit_data.npz un-ignore rule added to .gitignore**
- **Found during:** Task 2E (commit + push circuit_data.npz)
- **Issue:** `*.npz` rule in .gitignore blocked staging circuit_data.npz. File is intentionally gitignored in general, but the canonical file must be tracked so the cluster can git pull the mask-augmented version (otherwise flock guard triggers full retrain without mask key).
- **Fix:** Added `!/data/processed/rnn_behav/circuit_data.npz` un-ignore line with comment explaining the cluster pull requirement.
- **Files modified:** .gitignore (separate commit from backup gitignore entry)
- **Committed in:** 715b24f (Task 1 data commit includes the .gitignore change)

**2. [Rule 3 - Blocking] SSH to M3 not available from local machine (Tasks 1-2 → Task 3 gap)**
- **Found during:** Task 2E step 5 (cluster sweep submission)
- **Issue:** SSH to aman0087@m3.massive.org.au returned "Permission denied" — Monash VPN or interactive auth required from local machine.
- **Fix:** Created cluster/logs/n_latent_sweep_masked_jobs.txt as placeholder with submission instructions. All code changes committed and pushed to origin/main. Cluster submitted manually via M3 SSH + autopush. Task 3 ran 3 days later (2026-04-29) after autopush delivered artifacts via merge commits (ab9bccd, 0d29383).
- **Impact:** Not a problem — Tasks 1-2 and Task 3 are designed to be decoupled by autopush. The 3-day gap between Tasks 1-2 (2026-04-26) and Task 3 (2026-04-29) is the expected cluster latency pattern.

---

**Total deviations:** 2 (1 missing critical auto-fixed, 1 blocking workflow delay — not a code issue)
**Impact on plan:** Both handled correctly. No scope creep. SSH delay is a pre-existing cluster access constraint.

## Issues Encountered

- `import copy` was unused in circuit_inference.py (pre-existing lint issue, not introduced here). Not fixed to avoid scope creep on existing pre-existing lint backlog.
- Smoke test invariant_subspace_corr=0.065 on 5 inits / 50 epochs is expected low — smoke test is not a quality check, just an end-to-end correctness check. Cluster with 100 inits / 500 epochs is the quality run.
- 03-06-SUMMARY.md modification and `aggregate_per_context.py` / `per_context_results.json` (03-06 artifacts) were inadvertently staged with the Task 3 commit (f4574c8). This is a minor commit organization issue; the data is correct and both plans' artifacts are properly in the repo.

## Next Phase Readiness

- `wave_a_masked_selection.json` is written: chosen_rank=12, corr=0.5699, crossed_85_threshold=false, delta_vs_wave_a=-0.2134.
- **03-07 (shorter-T regen):** WILL RUN per conditional skip rule (crossed_85=false). Strong prior against improvement given negative delta, but must execute to complete the Phase 3.1 evidence base for 03-08.
- **03-08 (writeup re-commit):** Reads wave_a_masked_selection.json. Story direction pre-positioned toward STORY_1 (method/data limit). Checkpoint before Phase 4 unblock.
- **03-06 (per-context diagnostic):** Independent, can proceed in parallel. Task 1 complete. Task 2 (aggregation) awaits its own autopush.

---
*Phase: 03-latent-circuit-inference*
*Plan 05 — COMPLETE (all 3 tasks done, cluster sweep complete, aggregator committed)*
*Completed: 2026-04-29*
