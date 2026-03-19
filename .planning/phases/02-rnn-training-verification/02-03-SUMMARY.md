---
phase: 02-rnn-training-verification
plan: "03"
subsystem: training
tags: [pytorch, neurogym, context-decision-making, hidden-states, latent-circuit, actor-critic, rnn]

# Dependency graph
requires:
  - phase: 02-02
    provides: extract_behavior_with_hidden() — records hidden states per timestep with NaN padding
  - phase: 02-01
    provides: SingleContextDecisionMakingWrapper + neurogym installation in actinf-py-scripts env
  - phase: 01-01
    provides: ActorCritic canonical model at src/nn4psych/models/actor_critic.py

provides:
  - scripts/training/train_context_dm.py — full training + extraction pipeline for ContextDecisionMaking-v0
  - data/processed/rnn_behav/hidden_context_dm.npy — hidden state array (n_trials, max_T, 64)
  - data/processed/rnn_behav/trial_lengths_context_dm.npy — trial length array (n_trials,) int32
  - data/processed/rnn_behav/model_context_dm.pth — trained ActorCritic weights
  - data/processed/rnn_behav/metadata.json — model provenance and task metadata

affects:
  - 03-latent-circuit-inference (primary consumer of hidden_context_dm.npy)
  - any future training scripts that train on neurogym tasks

# Tech tracking
tech-stack:
  added: []
  patterns:
    - configure_cpu_threads() called at start of main() in all training scripts
    - Critic values detached via .item() before TD advantage computation (prevents double-backward)
    - hx.detach() after optimizer.step() to break computation graph across rollout buffers
    - numpy int64 shape values cast to int() before json.dump() for JSON serialization safety
    - output directory created with Path.mkdir(parents=True, exist_ok=True)

key-files:
  created:
    - scripts/training/train_context_dm.py
    - data/processed/rnn_behav/metadata.json
    - data/processed/rnn_behav/model_context_dm.pth
  modified: []

key-decisions:
  - "ContextDecisionMaking trials never signal done=True within 1000 steps — all trials run to max_steps_per_trial limit; no NaN padding in hidden array when all trials have equal length"
  - "Context-DM obs_dim=5 confirmed at runtime: 1 fixation + 2 modalities x 2 ring units (1 + 2*dim_ring with dim_ring=2)"
  - "input_dim=7: obs_dim=5 + context_dim=1 (single task) + reward=1"
  - "Actor loss uses detached advantages (no stop-gradient needed on actor side since advantages are plain tensors)"

patterns-established:
  - "Detach critic buffer values via .item() before advantage GAE computation — prevents RuntimeError: backward through graph a second time"
  - "Detach hx after optimizer.step() within rollout loop — required when rollout buffer is smaller than trial length"
  - "Cast numpy int64 to int() for JSON metadata — all shape values from .shape[] are numpy integers"

# Metrics
duration: 15min
completed: 2026-03-19
---

# Phase 2 Plan 03: Context-DM Training + Hidden State Extraction Summary

**Single-task ActorCritic trained on ContextDecisionMaking-v0 with hidden states saved as (20, 1000, 64) npy array for Phase 3 latent circuit inference**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-19T12:23:44Z
- **Completed:** 2026-03-19T12:38:00Z
- **Tasks:** 2
- **Files modified:** 3 (script created + 2 bug fixes applied)

## Accomplishments

- Created `scripts/training/train_context_dm.py`: full pipeline training ActorCritic on ContextDecisionMaking-v0 and extracting hidden states via `extract_behavior_with_hidden()`
- Training ran to completion (10 epochs x 20 trials); reward improved from ~3.4 to ~5.6
- Hidden states saved to `data/processed/rnn_behav/hidden_context_dm.npy` with shape (20, 1000, 64)
- All Phase 2 success criteria verified: neurogym available, obs_dim=5, hidden array is 3D with correct hidden_dim

## Task Commits

Each task was committed atomically:

1. **Task 1: Create train_context_dm.py** - `2c6ab3e` (feat)
2. **Task 2: Run training and verify outputs** - `cd9e30f` (feat — includes bug fixes)

**Plan metadata:** committed with SUMMARY.md update

## Files Created/Modified

- `scripts/training/train_context_dm.py` — Training script: trains ActorCritic on ContextDecisionMaking-v0, extracts hidden states, saves .npy + .pth + metadata.json
- `data/processed/rnn_behav/metadata.json` — Provenance: task, hidden_dim=64, obs_dim=5, action_dim=3, n_trials=20, seed=42, training/extraction hyperparameters
- `data/processed/rnn_behav/model_context_dm.pth` — Trained model weights (hidden_dim=64, input_dim=7, action_dim=3)
- `data/processed/rnn_behav/hidden_context_dm.npy` (gitignored) — Hidden state array shape (20, 1000, 64)
- `data/processed/rnn_behav/trial_lengths_context_dm.npy` (gitignored) — Trial lengths shape (20,) dtype int32

## Decisions Made

- **ContextDecisionMaking trial length behavior:** All trials run to max_steps_per_trial=1000 because the environment never signals done=True within that limit (consistent with DawTwoStep note in STATE.md). The hidden array has no NaN padding when all trials have equal length — this is correct and the downstream latent circuit code should account for it.
- **Single-task context vector:** `env.set_num_tasks(1)` makes context=[1.0] (length-1 one-hot), giving input_dim=7.
- **obs_dim=5 confirmed:** 1 fixation + 2 modalities x 2 ring units = 5 (formula 1 + 2*dim_ring, dim_ring=2).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed double-backward RuntimeError in advantage computation**
- **Found during:** Task 2 (run training)
- **Issue:** `buffer_values` stored `critic_value` tensors with grad_fn. When used as `next_value = values[t]` in advantage computation, the graph was freed after first `.backward()`, causing RuntimeError on the next rollout update.
- **Fix:** Extract float values via `.item()` into `values_detached` list before computing advantages. Added `hx = hx.detach()` after `optimizer.step()` to also break the graph for the hidden state.
- **Files modified:** `scripts/training/train_context_dm.py`
- **Verification:** Training ran to completion across all 10 epochs without error.
- **Committed in:** `cd9e30f` (Task 2 commit)

**2. [Rule 1 - Bug] Fixed JSON serialization TypeError for numpy int64 shape values**
- **Found during:** Task 2 (extract_and_save)
- **Issue:** `result['hidden'].shape[0]` and `result['hidden'].shape[1]` return `numpy.int64`, which raises `TypeError: Object of type int64 is not JSON serializable` in `json.dump()`.
- **Fix:** Cast `n_trials`, `max_T`, `obs_dim`, `action_dim`, `input_dim`, `hidden_dim`, `modality_context` to `int()` in the metadata dict.
- **Files modified:** `scripts/training/train_context_dm.py`
- **Verification:** `metadata.json` written successfully; `json.load()` reads it back correctly.
- **Committed in:** `cd9e30f` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes essential for correct execution. The advantage detach bug would cause training failure on any rollout buffer update after the first. The JSON fix prevents metadata from being saved. No scope creep.

## Issues Encountered

- `data/processed/rnn_behav/*.npy` files are gitignored (global `*.npy` rule in .gitignore). Committed `metadata.json` and `model_context_dm.pth` instead. The .npy arrays are reproducible by re-running the script.

## Next Phase Readiness

Phase 3 (Latent Circuit Inference) requires:
- `data/processed/rnn_behav/hidden_context_dm.npy` — shape (n_trials, max_T, hidden_dim) — EXISTS
- `data/processed/rnn_behav/trial_lengths_context_dm.npy` — shape (n_trials,) — EXISTS
- `scripts/training/train_context_dm.py` with `--skip_training --model_path` flag for reproducibility — EXISTS

Note for Phase 3: All context-DM trials run to max_steps (1000 timesteps). Latent circuit inference may want to filter to shorter sequences or set max_T in the rank selection. The engellab/latentcircuit repo compatibility check (noted as research flag in STATE.md) should happen before Phase 3 plan execution.

---
*Phase: 02-rnn-training-verification*
*Completed: 2026-03-19*
