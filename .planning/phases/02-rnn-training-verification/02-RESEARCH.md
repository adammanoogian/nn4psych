# Phase 2: RNN Training Verification - Research

**Researched:** 2026-03-18
**Domain:** PyTorch RNN training, NeuroGym task integration, hidden state extraction, convergence verification
**Confidence:** HIGH (all findings based on direct codebase inspection and live execution)

---

## Summary

Phase 2 has more latent problems than the phase description suggests. Direct code inspection and live execution reveal four blocking bugs in the existing training infrastructure before any new work (ContextDecisionMaking-v0 integration) can begin. The two existing training scripts each have structural problems: `train_rnn_canonical.py` is not importable as a module (runs top-level code), uses a private `ActorCritic` class instead of `src/nn4psych/models/actor_critic.py`, and saves to a `./model_params/` directory that must exist before training runs. `train_multitask.py` crashes when any NeuroGym task is used because it tries to access PIE-specific attributes (`bucket_positions`, `helicopter_positions`) on `NeurogymWrapper` instances. Additionally, neither training script extracts or saves hidden states — the `extract_behavior()` function in `behavior.py` only records behavioral data — so the success criterion requiring hidden states saved to `.npy` files requires a new function.

The ContextDecisionMaking-v0 observation space is also miscoded: the wrapper defaults say `obs_dim=3` ("1 + dim_ring") but the actual neurogym `_DMFamily` formula is `1 + 2*dim_ring`, giving `obs_dim=5` for `dim_ring=2`. This will cause a dimension mismatch when building the encoder layer in `MultiTaskActorCritic`.

neurogym is not installed in any local conda environment and is not in `pyproject.toml`. It must be added as an optional dependency and installed before NeuroGym tasks can run at all.

**Primary recommendation:** Fix the four bugs first (canonical script structure, multitask neurogym state crash, obs_dim mismatch, missing hidden state extraction), then install neurogym, then do task integration and training verification.

---

## Standard Stack

The stack is already determined by Phase 1 decisions. No new libraries needed for the core work.

### Core (already in environment)
| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| PyTorch | 2.9.1+cpu (base conda) | RNN training, actor-critic model | Working |
| numpy | - | Array ops, state storage | Working |
| scipy | - | Learning rate metrics | Working |
| gymnasium | >=0.28.0 | Environment interface (PIE uses this) | Working |

### Required but Missing
| Library | Purpose | Install Command |
|---------|---------|----------------|
| neurogym | NeuroGym task environments (ContextDecisionMaking-v0, DawTwoStep-v0) | `pip install neurogym` |

### Installation
```bash
# Add to pyproject.toml optional-dependencies:
# neurogym = ["neurogym>=0.0.2"]
pip install neurogym
```

**Note:** neurogym is not on PyPI as a stable release. The install is `pip install git+https://github.com/neurogym/neurogym` or a tagged release if available. The `NeurogymWrapper` already handles `ImportError` gracefully (`NEUROGYM_AVAILABLE = False`).

---

## Architecture Patterns

### Recommended Project Structure for Phase 2

No new top-level directories needed. Changes are within existing files and one new function in `behavior.py`.

```
nn4psych/
├── envs/
│   ├── neurogym_wrapper.py          # EXISTING — no changes needed (wrapper is correct)
│   └── pie_environment.py           # EXISTING — already has reset_epoch(), get_state_history()
├── src/nn4psych/
│   ├── models/
│   │   └── actor_critic.py          # EXISTING — canonical model, no changes needed
│   ├── analysis/
│   │   └── behavior.py              # ADD extract_behavior_with_hidden() function
│   └── training/
│       └── configs.py               # EXISTING — ExperimentConfig, no changes needed
├── scripts/training/
│   ├── train_rnn_canonical.py       # FIX: wrap in if __name__ == '__main__':
│   │                                #      use canonical ActorCritic from src/
│   │                                #      ensure output dirs exist (model_params/, figures/)
│   └── train_multitask.py           # FIX: neurogym state extraction crash (lines 481-498)
└── data/processed/rnn_behav/        # EXISTING — save hidden states here as .npy
```

### Pattern 1: Training Script Structure

**What:** Training scripts must be wrapped in `if __name__ == '__main__':` so they are importable without executing.
**When to use:** All scripts in `scripts/training/`.
**Current problem:** `train_rnn_canonical.py` executes argparse, training loop, and `torch.save()` at module scope. Importing it runs the full training run.

```python
# CORRECT pattern
def main():
    # ... all training code here ...

if __name__ == '__main__':
    main()
```

### Pattern 2: Canonical ActorCritic Usage in train_rnn_canonical.py

**What:** `train_rnn_canonical.py` defines its own `ActorCritic` class (lines 102-133) instead of importing from `src/nn4psych/models/actor_critic.py`.
**When to use:** Phase 2 should align the script with the canonical model.

```python
# CORRECT
from nn4psych.models.actor_critic import ActorCritic
# Remove the local class definition
```

**Note:** The local `ActorCritic` in the script uses `bias=False` hardcoded while `src/` version takes it as a parameter. They are architecturally identical; the script just predates the canonical version.

### Pattern 3: NeuroGym State Extraction in train_multitask.py

**What:** `train_epoch_interleaved()` (and block/trial variants) access PIE-specific attributes on all envs regardless of type.
**When to use:** Whenever training loop processes NeuroGym tasks.
**Current problem:** Lines 481-488 in `train_multitask.py`:
```python
states = np.array([
    env.trials,
    env.bucket_positions,   # AttributeError for NeurogymWrapper!
    env.bag_positions,
    env.helicopter_positions,
    env.hazard_triggers,
])
distances = np.abs(states[3] - states[1])
```

**Fix:** Add env-type dispatch:
```python
# Check env type for state extraction
if hasattr(env, 'bucket_positions'):
    # PIE environment
    states = np.array([env.trials, env.bucket_positions, env.bag_positions,
                       env.helicopter_positions, env.hazard_triggers])
    distances = np.abs(states[3] - states[1])
else:
    # NeurogymWrapper — use available attributes
    states = {'trials': env.trials, 'rewards': env.rewards_history,
              'trial_lengths': env.trial_lengths}
    distances = np.zeros(len(env.trials))  # Placeholder; NG tasks use reward as metric
```

### Pattern 4: Hidden State Extraction

**What:** A new `extract_behavior_with_hidden()` function that records `hx` at each timestep.
**When to use:** After training, to produce the arrays needed for Phase 3 (latent circuit inference).
**Current problem:** `extract_behavior()` in `behavior.py` does not collect hidden states.

The format required by downstream latent circuit inference (Phase 3) is:
- `hidden_states`: shape `(n_trials, max_timesteps, hidden_dim)` or a list of variable-length arrays (ragged)
- Recommend padding to max trial length with `np.nan` or 0, plus a `trial_lengths` array
- Save as `.npy` to `data/processed/rnn_behav/`

```python
def extract_behavior_with_hidden(
    model: ActorCritic,
    env,
    n_epochs: int = 100,
    n_trials: int = 200,
    device=None,
) -> dict:
    """
    Extract behavioral data AND hidden states for downstream analysis.

    Returns
    -------
    dict with keys:
        'states': list of epoch state tuples (from get_state_history())
        'hidden': np.ndarray, shape (n_epochs * n_trials, max_T, hidden_dim)
        'trial_lengths': np.ndarray, shape (n_epochs * n_trials,)
        'actions': list of action sequences per trial
        'rewards': list of reward sequences per trial
    """
```

**Save format for latent circuit inference:**
```python
# Save per condition
np.save('data/processed/rnn_behav/hidden_states_cp.npy', hidden_cp)   # (N, T, H)
np.save('data/processed/rnn_behav/hidden_states_ob.npy', hidden_ob)   # (N, T, H)
np.save('data/processed/rnn_behav/trial_lengths_cp.npy', lengths_cp)  # (N,)
np.save('data/processed/rnn_behav/trial_lengths_ob.npy', lengths_ob)  # (N,)
```

### Anti-Patterns to Avoid

- **Calling `torch.tensor()` on a list of tensors with requires_grad=True**: Use `torch.stack()` instead. Line 337 of `train_multitask.py` has this bug (confirmed by UserWarning at runtime). It does not break training but produces incorrect gradients in edge cases.
- **Saving model to non-existent directory**: `train_rnn_canonical.py` calls `torch.save()` to `./model_params/` which must already exist. Fix: `Path(model_path).parent.mkdir(parents=True, exist_ok=True)`.
- **Approximate obs_dim in configs**: The `NEUROGYM_TASK_DEFAULTS` and `TASK_REGISTRY` have `obs_dim=3` for ContextDecisionMaking-v0 which is wrong (actual is 5 for `dim_ring=2`). Always call `get_actual_task_dimensions()` at runtime — this is already implemented in `train_multitask.py` but the constants are wrong.
- **Hardcoded PIE-only context type in MultiTaskTrainer.train_trial()**: The method constructs context via `self.model.get_context(task_id, self.device)` (correct for MultiTaskActorCritic) but then checks `env.normalize_states()` which is on both PIE and NeurogymWrapper — this part is fine.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ContextDecisionMaking wrapper | Custom gym wrapper from scratch | `SingleContextDecisionMakingWrapper` (already exists) | Already implemented, tested in `tests/test_task_compatibility.py` |
| Hidden state padding | Custom padding loop | `np.full((N, max_T, H), np.nan)` + loop | Trivial and clear; ragged arrays are harder for downstream |
| Learning curve smoothing | Rolling average from scratch | `scipy.ndimage.uniform_filter1d` (already used in scripts) | Already imported and used |
| Model saving | Custom serialization | `torch.save(model.state_dict(), path)` | Correct PyTorch pattern already used |
| Convergence check | Stat test | Simple threshold: distance < 32 for PIE, mean reward > 0.5 for NeuroGym | Matches existing codebase convention |

---

## Common Pitfalls

### Pitfall 1: ContextDecisionMaking-v0 obs_dim Mismatch

**What goes wrong:** `MultiTaskActorCritic` builds an encoder with `in_features = obs_dim + context_dim + 1`. If `obs_dim=3` (wrong) but actual is 5, the forward pass crashes with a shape mismatch.
**Why it happens:** `NEUROGYM_TASK_DEFAULTS` and `TASK_REGISTRY` have `obs_dim: 3  # 1 + dim_ring` as a comment — but the formula for ContextDecisionMaking (which has TWO modalities) is `1 + 2*dim_ring = 5` for `dim_ring=2`.
**How to avoid:** Always use `get_actual_task_dimensions()` which instantiates a temp env. This function exists in `train_multitask.py` lines 994-1022 and is called by `create_task_specs()`. The bug is in the static defaults only; the runtime path is correct.
**Warning signs:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied` during first forward pass of a NeuroGym task.

### Pitfall 2: neurogym Not Installed

**What goes wrong:** `NEUROGYM_AVAILABLE = False` silently; all neurogym tasks fall back to `raise ImportError`. Training on NeuroGym tasks silently fails to even start.
**Why it happens:** `neurogym` is not in `pyproject.toml` and not installed in any local conda environment (confirmed by checking all conda envs).
**How to avoid:** Install before Phase 2 work: `pip install git+https://github.com/neurogym/neurogym`. Add as optional dep: `pip install -e ".[neurogym]"`.
**Warning signs:** `NEUROGYM_AVAILABLE: False` in training script startup output.

### Pitfall 3: train_rnn_canonical.py Cannot be Safely Imported or Tested

**What goes wrong:** Any attempt to `import scripts.training.train_rnn_canonical` triggers a full training run AND a `torch.save()` that fails because `./model_params/` does not exist.
**Why it happens:** All code is at module scope, no `if __name__ == '__main__':` guard.
**How to avoid:** Refactor into `main()` function before writing tests.
**Warning signs:** Running `python -m pytest` causes unexpected training runs.

### Pitfall 4: GAE Computation Bug (Tensor Detach Warning)

**What goes wrong:** `torch.tensor(advantages, ...)` where `advantages` is a list of tensors with `requires_grad=True` produces a UserWarning and may produce incorrect gradients.
**Why it happens:** Line 190 of `train_rnn_canonical.py` and line 337 of `train_multitask.py` both do this. The values `values[t]` come from model forward passes and carry grad.
**How to avoid:** Use `torch.stack([v.detach() for v in advantages])` or compute values without grad tracking.
**Warning signs:** `UserWarning: Converting a tensor with requires_grad=True to a scalar` printed during training.

### Pitfall 5: NeuroGym Trial Structure — done=True Within a Trial

**What goes wrong:** NeuroGym's `TrialEnv` can return `done=True` mid-trial (e.g., abort on fixation break). The training loop in `train_multitask.py` treats `done=True` as trial end and calls `env.reset()` for the next trial — this is correct behavior.
**Why it happens:** Not a bug, but trial lengths vary dramatically and can be very short (1-3 steps on abort), making rollout buffer fill unpredictably slow.
**How to avoid:** Use a large enough rollout buffer (>=50) and expect high variance in early NeuroGym training. Do not set `max_time` constraints that conflict with neurogym's internal timing.
**Warning signs:** Rollout buffer never fills (too few steps per trial), causing no gradient updates.

### Pitfall 6: NeurogymWrapper `done` After Reset Always Returns False

**What goes wrong:** `NeurogymWrapper.reset()` always returns `(obs, False)` — this is intentional. The `done` flag only becomes True from `step()`. But if a NeuroGym task internally marks the episode as done immediately on reset (some tasks do), the outer while loop never runs.
**Why it happens:** NeuroGym's `env.reset()` can configure internal state to be immediately terminal in some edge cases.
**How to avoid:** Add a step-count guard in trial loops: `while not done and steps < max_steps`.

---

## Code Examples

Verified patterns from direct codebase inspection:

### extract_behavior_with_hidden() Function Signature

```python
# Source: src/nn4psych/analysis/behavior.py (to be added)
def extract_behavior_with_hidden(
    model: ActorCritic,
    env,              # PIE_CP_OB_v2 or NeurogymWrapper
    n_epochs: int = 1,
    n_trials: int = 200,
    reset_memory: bool = True,
    preset_memory: float = 0.0,
    device: Optional[torch.device] = None,
) -> dict:
    """Returns dict with 'states', 'hidden' (N, T, H), 'trial_lengths' (N,), 'actions', 'rewards'."""
```

### PIE Training Convergence Check

```python
# Source: train_rnn_canonical.py lines 321-329, train_multitask.py lines 752-759
# Learning criterion: mean heli-bucket distance < 32
perf = np.mean(abs(all_states[epoch, :, 3] - all_states[epoch, :, 1]))  # heli - bucket
if perf < 32:  # Learning threshold for PIE tasks
    # Save checkpoint
```

### NeuroGym Task Convergence Check (NEW — not yet in codebase)

```python
# For DawTwoStep and ContextDecisionMaking, use mean reward as criterion
mean_reward = np.mean(env.rewards_history[-100:])  # Last 100 trials
if mean_reward > 0.5:  # Corresponds to >50% correct on binary choice tasks
    # Model has learned the task
```

### Correct State Construction for MultiTaskActorCritic

```python
# Source: train_multitask.py lines 372-378 (train_trial method)
norm_obs = env.normalize_states(next_obs)
context = self.model.get_context(task_id, self.device)  # (context_dim,) one-hot
next_state = torch.cat([
    torch.FloatTensor(norm_obs).to(self.device),
    context,
    torch.tensor([0.0], device=self.device),  # reward
])
next_state = next_state.unsqueeze(0).unsqueeze(0)  # (1, 1, obs+ctx+1)
```

### Directory Creation Before Saving

```python
# Fix for train_rnn_canonical.py save crash
from pathlib import Path
save_path = Path('data/processed/rnn_behav/hidden_cp.npy')
save_path.parent.mkdir(parents=True, exist_ok=True)
np.save(save_path, hidden_states_cp)
```

---

## ContextDecisionMaking-v0 Actual Specification

**Source:** neurogym GitHub `yang19.py` — `_DMFamily` class (confirmed via WebFetch)
**Confidence:** HIGH (verified against neurogym source)

| Property | Value | Notes |
|----------|-------|-------|
| Task class | `_DMFamily` with `w_mod=(1,0)` or `(0,1)` | Registered as `ContextDecisionMaking-v0` |
| `obs_dim` (dim_ring=2) | **5** | Formula: `1 + 2*dim_ring` = 1 fixation + 2 modalities × 2 ring units |
| `action_dim` (dim_ring=2) | **3** | Formula: `1 + dim_ring` = fixation + 2 choices |
| Coded default in wrapper | obs_dim=3 (WRONG) | Must use `get_actual_task_dimensions()` at runtime |
| Reward: correct | +1.0 | |
| Reward: fail | 0.0 | |
| Reward: abort (fixation break) | -0.1 | |
| Trial stages | Fixation → Stimulus → Delay → Decision | Multi-period, variable length |

The `SingleContextDecisionMakingWrapper` passes `env_name='ContextDecisionMaking-v0'` to `ngym.make()` — this is correct. The `modality_context` parameter controls which modality's coherence determines the correct answer.

---

## Learning Verification Criteria

### PIE Tasks (change-point, oddball)
- **Metric:** Mean heli-bucket distance across trials
- **Threshold:** `< 32` (1/10 of environment range 0-301)
- **Source:** `train_rnn_canonical.py` line 321, `train_multitask.py` line 757
- **Epochs needed:** Typically 100-1000 epochs. The existing script uses `args.epochs` (default 10 — too low for real convergence). For verification, run 100+ epochs.
- **Confidence:** HIGH (this threshold is in existing working code)

### NeuroGym Tasks (DawTwoStep, ContextDecisionMaking)
- **Metric:** Mean trial reward over last 100 trials
- **Threshold:** `> 0.5` for binary choice tasks (above chance)
- **Alternative:** `reward_history[-1] > 0` (any positive reward in last trial)
- **Epochs needed:** NeuroGym tasks typically converge in 50-500 epochs at rollout_size=50
- **Confidence:** MEDIUM (threshold derived from task structure; no existing code uses it)

### Visual Verification
- Training curves should be non-flat (reward increasing) and non-diverging (reward not going to -inf)
- For PIE: plot `epoch_perf[:, :, 0]` (returns) vs epoch — should show upward trend
- For NeuroGym: plot `rewards_history` over training — should stabilize above chance

---

## Hidden State Extraction — Format for Latent Circuit Inference

Phase 3 (Latent Circuit Inference) requires fitting Q, w_rec, w_in, w_out from hidden states. Based on the ROADMAP, the latent inference algorithm needs:

1. **Activity matrix X:** shape `(T_total, hidden_dim)` where T_total = total timesteps across all trials
2. **Trial index array:** `(T_total,)` indicating which trial each timestep belongs to
3. **Task condition label:** `(T_total,)` or `(n_trials,)` — which condition (CP vs OB vs context-DM)

**Recommended save format:**
```
data/processed/rnn_behav/
├── hidden_cp.npy          # (n_trials * max_T, hidden_dim) or ragged as object array
├── hidden_ob.npy          # same
├── hidden_context_dm.npy  # same
├── trial_lengths_cp.npy   # (n_trials,) — lengths of each trial
├── trial_lengths_ob.npy
├── trial_lengths_context_dm.npy
├── actions_cp.npy         # (n_trials, max_T) — action taken at each step
├── rewards_cp.npy         # (n_trials, max_T)
└── metadata.json          # model_path, hidden_dim, n_trials, condition
```

**Shape convention:** `hidden_states[trial, timestep, unit]` — trial-major order matches Phase 3 fitting convention (inferred from ROADMAP description of 100-init ensemble fitting).

---

## Existing Test Coverage

### Tests that validate training prerequisites (PASSING based on direct code inspection)

| Test | File | What It Covers |
|------|------|----------------|
| `TestActorCriticForward` | `tests/test_actor_critic.py` | Forward pass shapes, gradient flow |
| `TestModelEnvironmentLoop.test_model_completes_epoch` | `validation/test_model_environment.py` | End-to-end model-env loop |
| `TestModelEnvironmentLoop.test_extract_behavior_runs` | `validation/test_model_environment.py` | extract_behavior returns 3 epochs |
| `TestPIEChangepoint.test_complete_trial` | `tests/test_task_compatibility.py` | PIE trial with MultiTaskActorCritic |
| `TestMultiTaskPIEOnly.test_run_trials_both_tasks` | `tests/test_task_compatibility.py` | Both PIE tasks with shared model |

### Tests that need NeuroGym (SKIPPED without neurogym)

| Test | File | What It Covers |
|------|------|----------------|
| `TestNeurogymDawTwoStep` | `tests/test_task_compatibility.py` | DawTwoStep obs/act shapes, trial completion |
| `TestNeurogymContextDM` | `tests/test_task_compatibility.py` | ContextDM obs/act shapes, trial completion |
| `TestMultiTaskMixed` | `tests/test_task_compatibility.py` | Mixed PIE + DawTwoStep model |

### Tests NOT YET WRITTEN (needed for Phase 2 success criteria)

1. `test_training_convergence_pie` — run 50 epochs, assert distance < 32
2. `test_extract_behavior_with_hidden` — assert hidden array shape is `(n_epochs, n_trials, max_T, hidden_dim)` or equivalent
3. `test_hidden_states_save_to_npy` — assert files exist at `data/processed/rnn_behav/`
4. `test_neurogym_training_no_crash` — run 5 epochs of multitask with NeuroGym task, assert no AttributeError

---

## Config Changes Needed Per Task Type

### PIE tasks (no changes needed)

`ExperimentConfig` defaults are already correct:
```python
ModelConfig(input_dim=9, hidden_dim=64, action_dim=3)
# input_dim = 6 obs + 2 context (one-hot for 2 tasks) + 1 reward = 9
```

For single-task training with `ActorCritic` (not `MultiTaskActorCritic`):
```python
# From train_rnn_canonical.py
input_dim = 6 + 3  # 6 obs + 2 context + 1 reward = 9 total
```

### NeuroGym DawTwoStep-v0

Actual obs_dim must be queried at runtime via `get_actual_task_dimensions()`. The config default `obs_dim=4` in `TASK_REGISTRY` is an estimate. For the `MultiTaskActorCritic`, the encoder handles the variable input size automatically.

### NeuroGym ContextDecisionMaking-v0 (dim_ring=2)

```python
# CORRECT values (not the coded defaults):
obs_dim = 5   # 1 + 2*dim_ring = 1 + 2*2 = 5
action_dim = 3  # 1 + dim_ring = 1 + 2 = 3
```

The `TASK_REGISTRY` entry for `single-context-dm` has `obs_dim=3` which is WRONG. Must update or always call `get_actual_task_dimensions()` (which creates a temp env and reads the actual value).

The `MultiTaskConfig` in `train_multitask.py` uses `create_task_specs()` → `get_actual_task_dimensions()`, so the runtime path is correct despite the wrong static default. The static default only affects `TASK_REGISTRY` lookups that don't go through the runtime check.

---

## State of the Art

| Old Approach | Current Approach | Impact on Phase 2 |
|--------------|-----------------|-------------------|
| Top-level script code | `if __name__ == '__main__': main()` | train_rnn_canonical.py needs this fix |
| PIE-only training | MultiTaskActorCritic with shared RNN | Already built; has NeuroGym crash to fix |
| `torch.tensor(grad_tensor_list)` | `torch.stack([t.detach() for t in list])` | Minor bug to fix in both training scripts |
| Behavioral data only | Behavioral + hidden state arrays | New `extract_behavior_with_hidden()` needed |

---

## Open Questions

1. **DawTwoStep-v0 actual obs/action dimensions**
   - What we know: `NEUROGYM_TASK_DEFAULTS` says `obs_dim=4, action_dim=3` as estimates
   - What's unclear: Actual values depend on neurogym version and dt parameter
   - Recommendation: Always use `get_actual_task_dimensions()` (creates temp env); update static defaults after confirming

2. **neurogym install method**
   - What we know: Not on conda-forge or standard PyPI channel; GitHub repo available
   - What's unclear: Whether `pip install neurogym` works from PyPI or needs GitHub URL
   - Recommendation: Try `pip install neurogym` first; fall back to `pip install git+https://github.com/neurogym/neurogym`

3. **ContextDecisionMaking task name in neurogym**
   - What we know: The wrapper uses `env_name='ContextDecisionMaking-v0'`; prior Phase 1 work noted "NeuroGym ContextDecisionMaking-v0 task exists and loads"
   - What's unclear: Whether the registration name is exactly `ContextDecisionMaking-v0` or if it's in a subpackage
   - Recommendation: Verify with `import neurogym; print(list(neurogym.envs.ALL_ENVS.keys()))` after install

4. **Hidden state format for Phase 3 latent circuit inference**
   - What we know: Phase 3 ROADMAP says "fit Q, w_rec, w_in, w_out from context-DM RNN hidden states with 100-init ensemble"
   - What's unclear: Whether Phase 3 expects flat `(T_total, H)` or trial-structured `(N, T, H)`
   - Recommendation: Save both formats; primary as `(N, T, H)` with padding, secondary as flat with trial index

5. **Convergence epochs for NeuroGym tasks**
   - What we know: PIE converges in 100-1000 epochs; NeuroGym tasks have different reward scales (+1/-0.1)
   - What's unclear: How many epochs needed for DawTwoStep and ContextDecisionMaking
   - Recommendation: Run 200 epochs as a smoke test; convergence is not required for Phase 2, only "runs to completion without error"

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection: `scripts/training/train_rnn_canonical.py` — lines 102-133 (local ActorCritic), no `__main__` guard
- Direct codebase inspection: `scripts/training/train_multitask.py` — lines 481-498 (PIE-only state access), line 337 (GAE bug)
- Direct codebase inspection: `envs/neurogym_wrapper.py` — `SingleContextDecisionMakingWrapper` uses `env_name='ContextDecisionMaking-v0'`
- Direct codebase inspection: `src/nn4psych/training/configs.py` — `obs_dim=3` for `ContextDecisionMaking-v0` in `TASK_REGISTRY`
- Live execution (conda base with torch 2.9.1): `extract_behavior()` returns `(trials, buckets, bags, helis, hazards)` tuples; no hidden states
- Live execution: `MultiTaskTrainer` runs end-to-end for PIE tasks in 2 epochs
- Live execution: `train_rnn_canonical.py` import triggers training and crashes on missing `./model_params/` directory
- neurogym GitHub `yang19.py`: `_DMFamily` obs formula = `1 + 2*dim_ring`, action = `1 + dim_ring` (via WebFetch)
- neurogym GitHub native/perceptualdecisionmaking.py: `obs_dim = 1 + dim_ring` for PerceptualDecisionMaking (via WebFetch)

### Secondary (MEDIUM confidence)
- `tests/test_task_compatibility.py` — tests skip when `NEUROGYM_AVAILABLE=False`; neurogym confirmed not installed
- `data/processed/rnn_behav/model_params_101000/` — existing `.pkl` files show behavioral data was previously saved in pickle format, not `.npy`; phase 2 success criterion requires `.npy`

### Tertiary (LOW confidence)
- Neurogym install method: `pip install neurogym` vs GitHub URL — not verified (neurogym not in current env)
- DawTwoStep-v0 actual obs/action dims — not confirmed (needs live env to verify)
- Epochs needed for NeuroGym convergence — estimated from task difficulty, not measured

---

## Metadata

**Confidence breakdown:**
- Training script bugs: HIGH — confirmed by live execution and code inspection
- ContextDecisionMaking obs_dim: HIGH — confirmed by neurogym source (WebFetch)
- Hidden state extraction format: MEDIUM — format is straightforward; downstream Phase 3 spec not yet defined
- NeuroGym convergence criteria: MEDIUM — derived from task structure; not from existing code
- neurogym install: LOW — package not installed anywhere; method unverified

**Research date:** 2026-03-18
**Valid until:** 2026-04-18 (neurogym changes slowly; PyTorch API stable)
