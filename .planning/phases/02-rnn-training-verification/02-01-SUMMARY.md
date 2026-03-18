---
phase: 02
plan: 01
subsystem: training-infrastructure
tags: [pytorch, actor-critic, neurogym, bug-fix, training-scripts]

dependency-graph:
  requires: [01-03]
  provides: [importable-training-scripts, env-type-dispatch, correct-obs-dims, neurogym-optional-dep]
  affects: [02-02, 02-03]

tech-stack:
  added: []
  patterns:
    - hasattr env-type dispatch (PIE vs NeurogymWrapper state extraction)
    - torch.stack with detach for GAE tensor construction
    - if __name__ == '__main__' guard on training scripts

key-files:
  created: []
  modified:
    - scripts/training/train_rnn_canonical.py
    - scripts/training/train_multitask.py
    - src/nn4psych/training/configs.py
    - envs/neurogym_wrapper.py
    - pyproject.toml

decisions:
  - id: evaluate-method-also-guarded
    description: "Also applied hasattr guard to evaluate() method — 4 sites total, not 3 as planned"
    rationale: "evaluate() had identical unguarded PIE-only state access that would crash on NeurogymWrapper envs"
    impact: "evaluate() now works with both env types"

metrics:
  duration: "12 min"
  completed: "2026-03-18"
---

# Phase 2 Plan 1: Training Script Bug Fixes Summary

**One-liner:** Fixed 4 blocking bugs in training scripts (main guard, NeuroGym state crash, GAE tensor, local ActorCritic) and corrected ContextDecisionMaking obs_dim from 3 to 5.

## Objective

Fix bugs so both training scripts can run without crashing across all three task types (PIE, NeuroGym, context-DM).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Refactor train_rnn_canonical.py | 8a695c9 | scripts/training/train_rnn_canonical.py |
| 2 | Fix train_multitask.py — NeuroGym crash + GAE + obs_dim | 91091c7 | scripts/training/train_multitask.py, src/nn4psych/training/configs.py, envs/neurogym_wrapper.py |
| 3 | Add neurogym optional dependency to pyproject.toml | 9427d04 | pyproject.toml |

## Bugs Fixed

### Bug 1 — No main guard in train_rnn_canonical.py
- **Impact:** Importing the module would immediately execute argparse and training
- **Fix:** Wrapped all execution code in `main()` + `if __name__ == '__main__': main()`
- **Verification:** `python -c "from scripts.training import train_rnn_canonical"` returns without output

### Bug 2 — Local ActorCritic class in train_rnn_canonical.py
- **Impact:** Training diverged from the canonical model in `src/nn4psych/models/actor_critic.py`; two sources of truth
- **Fix:** Deleted local class (lines 102-134); added `from nn4psych.models.actor_critic import ActorCritic`
- **Verification:** No `class ActorCritic` in train_rnn_canonical.py; canonical link confirmed

### Bug 3 — NeuroGym state extraction crash in train_multitask.py
- **Impact:** Calling `env.bucket_positions` on a NeurogymWrapper env raises `AttributeError`
- **Fix:** Added `hasattr(env, 'bucket_positions')` dispatch in 4 locations:
  - `train_epoch_interleaved()` (line ~487)
  - `train_epoch_trial_interleaved()` (line ~604)
  - `train_epoch_block_interleaved()` (line ~704)
  - `evaluate()` (line ~879) — extra site not in plan, auto-fixed as Rule 1 bug
- **Verification:** `grep -c "hasattr(env, 'bucket_positions')" train_multitask.py` returns 4

### Bug 4 — GAE torch.tensor() on grad tensors (both scripts)
- **Impact:** `torch.tensor(advantages, ...)` raises UserWarning about copying grad tensors; gradients silently detached
- **Fix:** Replaced with `torch.stack([a.detach() if isinstance(a, torch.Tensor) else torch.tensor(a) for a in advantages])`
- **Verification:** `grep "detach" train_rnn_canonical.py train_multitask.py` matches in both files

### Bug 5 — Incorrect obs_dim=3 for ContextDecisionMaking (3 locations)
- **Impact:** Model input dimension mismatch at runtime; training would fail with shape errors
- **Fix:** Corrected to `obs_dim=5` (formula: 1 fixation + 2 modalities × 2 ring units = 1 + 2*dim_ring)
  - `TASK_REGISTRY['single-context-dm']` in configs.py
  - `NEUROGYM_TASK_DEFAULTS['ContextDecisionMaking-v0']` in neurogym_wrapper.py
  - `AVAILABLE_TASKS['single-context-dm']` in train_multitask.py
- **Verification:** `grep "obs_dim=5" configs.py` and `grep "'obs_dim': 5" neurogym_wrapper.py`

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Also guarded `evaluate()` method (4th site) | evaluate() had identical unguarded PIE-only state access that would crash on NeurogymWrapper. Plan mentioned 3 training methods but evaluate had the same pattern — auto-fixed as Rule 1 bug. |
| train_with_rollouts keeps module-level signature but adds explicit params | Avoided closure leakage; hidden_dim, device, reset_memory, bias now passed explicitly. Less disruptive than nesting inside main(). |
| obs_dim formula documented as 1 + 2*dim_ring not 1 + dim_ring | dim_ring=2 means 2 ring units per modality, 2 modalities = 4 sensory dims + 1 fixation = 5 total. Original comment was wrong. |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Guarded evaluate() method in addition to the 3 planned training methods**

- **Found during:** Task 2 while verifying all bucket_positions accesses
- **Issue:** `evaluate()` had identical unguarded `env.bucket_positions` access; would crash when called on any NeuroGym task
- **Fix:** Added same hasattr dispatch block to evaluate() — 4 total guards vs 3 planned
- **Files modified:** scripts/training/train_multitask.py
- **Commit:** 91091c7

## Next Phase Readiness

- Both training scripts are importable and structurally correct
- All three task types can be instantiated without crashes
- obs_dim values are consistent across TASK_REGISTRY, NEUROGYM_TASK_DEFAULTS, and AVAILABLE_TASKS
- neurogym is installed in the project conda env (actinf-py-scripts) and available as optional dep
- Ready for 02-02 (smoke-test training runs) and 02-03 (validation tests)
