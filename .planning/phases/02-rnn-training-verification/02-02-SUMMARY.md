---
phase: 02-rnn-training-verification
plan: 02
subsystem: training
tags: [pytorch, rnn, actor-critic, neurogym, hidden-states, behavior-extraction, multitask]

# Dependency graph
requires:
  - phase: 02-01
    provides: training script bug fixes (hasattr guards, GAE, obs_dim correction)
  - phase: 01-03
    provides: NeurogymWrapper with reset_epoch() public API

provides:
  - extract_behavior_with_hidden() function in behavior.py returning (N, max_T, H) hidden state arrays
  - Verified PIE training end-to-end via train_rnn_canonical.py
  - Verified multi-task PIE training end-to-end via train_multitask.py
  - Verified multi-task training with NeuroGym (DawTwoStep) end-to-end
  - plot_lrs() guards for small-epoch smoke tests

affects:
  - 02-03 (model persistence and loading)
  - 03-latent-circuit-inference (consumes hidden state arrays of shape N x max_T x H)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Hidden states padded with NaN (not zeros) so downstream code can distinguish padding from real zero activations"
    - "Trial-major hidden state layout: (n_trials, max_T, hidden_dim)"
    - "Deterministic action selection (argmax) in extraction functions, not stochastic sampling"
    - "extract_behavior_with_hidden uses only public env API: reset_epoch, reset, step, normalize_states, context, get_state_history"

key-files:
  created: []
  modified:
    - src/nn4psych/analysis/behavior.py
    - scripts/training/train_rnn_canonical.py

key-decisions:
  - "extract_behavior_with_hidden uses argmax (deterministic) not sampling — consistent with extract_behavior()"
  - "NaN padding chosen over zero padding for hidden states to allow masking in downstream analysis"
  - "plot_lrs() edge case fixed with max(0, id-gap):id+1 slice and empty-array guards — smoke tests now clean"

patterns-established:
  - "Behavior extraction pattern: both PIE and NeurogymWrapper share the same extract_behavior_with_hidden interface"
  - "Smoke test pattern: --epochs 3 --trials 10 --maxt 30 is sufficient to validate end-to-end training"

# Metrics
duration: 15min
completed: 2026-03-19
---

# Phase 2 Plan 2: RNN Behavior + Hidden State Extraction Summary

**extract_behavior_with_hidden() delivering (N, max_T, hidden_dim) NaN-padded arrays from PIE and NeurogymWrapper envs; all training pipelines verified end-to-end**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-19T12:13:59Z
- **Completed:** 2026-03-19T12:29:00Z
- **Tasks:** 2 (Task 1 pre-committed at 55e5b6d; Task 2 executed this session)
- **Files modified:** 2

## Accomplishments

- `extract_behavior_with_hidden()` in `src/nn4psych/analysis/behavior.py` returns `hidden` ndarray of shape `(n_total_trials, max_T, hidden_dim)` with NaN padding — ready for Phase 3 latent circuit fitting
- PIE canonical training (`train_rnn_canonical.py`) runs to completion (3 epochs, 10 trials, maxt 30)
- Multi-task training with PIE-only tasks (change-point + oddball) completes cleanly
- Multi-task training with NeuroGym task (change-point + daw-two-step) completes without AttributeError
- `extract_behavior_with_hidden` verified on both PIE (`hidden shape: (10, 4, 64)`) and NeurogymWrapper (`hidden shape: (5, 1000, 64)`)

## Task Commits

1. **Task 1: Implement extract_behavior_with_hidden()** - `55e5b6d` (feat) — pre-committed
2. **Task 2: Verify training end-to-end + fix plot_lrs** - `177e843` (fix)

**Plan metadata:** (this commit)

## Files Created/Modified

- `src/nn4psych/analysis/behavior.py` — Added `extract_behavior_with_hidden()` function (Task 1, pre-committed)
- `scripts/training/train_rnn_canonical.py` — Fixed `plot_lrs()` to guard against empty arrays in small-epoch runs

## Decisions Made

- `extract_behavior_with_hidden()` uses `argmax` (deterministic) rather than sampling, matching `extract_behavior()` convention — keeps extraction reproducible for analysis
- NaN padding chosen over zero padding so downstream code (latent circuit fit) can distinguish real zero activations from padded timesteps
- `plot_lrs()` fix uses `max(0, id-gap):id+1` slice and empty-array filtering — minimal change that makes smoke tests with few epochs fully clean (previously acceptable to crash per plan; now clean)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed plot_lrs() crash on empty state slice (small epoch counts)**

- **Found during:** Task 2, Verification A (train_rnn_canonical.py with 3 epochs)
- **Issue:** `plot_lrs(all_states[id-gap:id])` with `id=0` and `gap=100` produces empty slice; `np.concatenate([])` raises `ValueError: need at least one array to concatenate`. Also `id-gap:id` excluded `id` itself, leaving very few epochs even for `id>100`.
- **Fix:** (1) Early return `[empty, empty], [empty, empty], [0.0, 0.0]` when `epochs==0`; (2) filter empty per-trial arrays before concatenating; (3) changed slice to `max(0, id-gap):id+1` to include the target epoch; (4) guarded `window_size = max(1, int(len(lrss[c])*0.2))` for `uniform_filter1d`.
- **Files modified:** `scripts/training/train_rnn_canonical.py`
- **Verification:** `train_rnn_canonical.py --epochs 3 --trials 10 --maxt 30` completes with "Fig saved" output
- **Committed in:** `177e843`

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Fix was optional per plan ("acceptable to crash in plotting") but improves smoke test clarity. No scope creep.

## Issues Encountered

- `np.trapz` deprecation warning (use `np.trapezoid`) — not fixed as it is a warning only, not an error; deferred to future cleanup pass

## Next Phase Readiness

- `extract_behavior_with_hidden()` is ready for Phase 3 consumption: returns `(n_trials, max_T, hidden_dim)` NaN-padded array
- All training pipelines verified: PIE canonical, multi-task PIE, multi-task with NeuroGym
- Blocker: none
- Note for Phase 3: `NeurogymWrapper` DawTwoStep produces `max_T=1000` (long trials at dt=100ms) — may need trial length filtering before latent circuit fitting

---
*Phase: 02-rnn-training-verification*
*Completed: 2026-03-19*
