# Phase 3 Research: Latent Circuit Inference

**Researched:** 2026-03-19
**Sources:** latentcircuit/Tutorial.ipynb, latentcircuit/latent_net.py, latentcircuit/net.py, latentcircuit/connectivity.py, latentcircuit/Tasks/SiegelMillerTask.py, latentcircuit/Tasks/ManteTask.py, src/nn4psych/models/actor_critic.py, src/nn4psych/analysis/behavior.py, envs/neurogym_wrapper.py, scripts/training/train_context_dm.py, neurogym contextdecisionmaking.py

---

## 1. Key Findings (What the planner needs to know)

### 1.1 LatentNet is a thin PyTorch module — it fits on top of existing hidden states

`LatentNet` (latent_net.py) is a standalone `nn.Module`. It learns four things jointly:
- `Q` — an (n x N) orthonormal embedding matrix (n = latent rank, N = full RNN hidden dim). Parameterized via Cayley transform of a free matrix `a`. Not a weight in the usual sense — it is recomputed from `a` after every gradient step.
- `w_rec` — (n x n) recurrent weights of the latent circuit
- `w_in` — (input_size x n) input weights, masked to diagonal
- `w_out` — (n x output_size) output weights, masked to diagonal

The loss has two terms:
- `mse_z`: mean squared error between `w_out(x)` and task target outputs `z`
- `l_y * nmse_y`: normalized MSE between `x @ Q` and full RNN hidden states `y` (reconstruction term)

These are minimized jointly. The key point: LatentNet needs both the task outputs `z` AND the RNN hidden states `y` to fit.

### 1.2 Tutorial uses n=8 latent units for a 50-unit RNN on Siegel/Mante context-DM

From Tutorial.ipynb:
```python
latent_net = LatentNet(n=8, N=net.n, input_size=6, n_trials=u.shape[0], sigma_rec=0.15)
```
- Their RNN: N=50 units
- Their latent circuit: n=8 units
- Input channels: 6 (their 6-dim Mante/Siegel task format)

For our case (N=64 hidden units), a reasonable starting rank is **n=8** following the same ratio. The tutorial does not specify an exact universally required rank — it is a hyperparameter that trades off expressiveness vs. interpretability. The invariant subspace correlation test (Q^T W_rec Q vs w_rec correlation >= 0.85) is the empirical criterion for whether the chosen rank is sufficient.

**Confidence:** MEDIUM — directly from Tutorial.ipynb code, but rank for our specific N=64 / 7-dim input case requires empirical validation.

### 1.3 The fitting procedure uses 500 epochs with Adam, lr=0.02, l_y=1, weight_decay=0.001

From Tutorial.ipynb:
```python
loss_history = latent_net.fit(u.detach(), z.detach(), y.detach(),
                              epochs=500, lr=.02, l_y=1, weight_decay=0.001)
```
Internal batch size: 128 (hardcoded in `fit()`).

For 100-initialization ensemble: run `fit()` 100 times with fresh `LatentNet` instances, record final `nmse_y` from each, select the initialization with lowest `nmse_y` as best solution.

**Confidence:** HIGH — directly from source code.

### 1.4 The three validation checks from the tutorial

The tutorial demonstrates exactly three validation approaches:

1. **Qx vs y scatter** (activity level, full space): Compute `latent_net(u) @ latent_net.q` vs `net(u)`. Good agreement = circuit captures full-space dynamics.

2. **x vs Q^T y scatter** (activity level, latent space): Compute `net(u) @ latent_net.q.t()` vs `latent_net(u)`. Good agreement = circuit explains variance in the latent subspace.

3. **w_rec vs Q^T W_rec Q heatmap** (connectivity level / invariant subspace): Compare `latent_net.recurrent_layer.weight.data` with `q @ W_rec @ q.T`. Visual agreement + correlation value.

The invariant subspace condition from the roadmap maps to check #3. The activity-level check maps to checks #1 and #2.

**Confidence:** HIGH — directly from Tutorial.ipynb cells.

### 1.5 The big problem: we do NOT have u and z for our ActorCritic

This is the central adapter challenge. The latentcircuit tutorial workflow:
```
generate_trials() → u, z, mask, conditions
net(u) → y   (simulate RNN to get hidden states)
latent_net.fit(u, z, y)
```

Our workflow from Phase 2:
```
extract_behavior_with_hidden() → hidden states (padded NaN array)
```

Our `extract_behavior_with_hidden` captures `hidden` (shape: `n_trials x max_T x 64`) but does NOT capture:
- `u`: the raw stimulus input at each timestep — the ActorCritic's actual input tensor `x = [norm_obs, context, reward]`
- `z`: the task target output — in our RL setting this would need to be reconstructed as action probabilities or actor logits

**This requires re-running the model** (or modifying `extract_behavior_with_hidden`) to also capture the input sequence and a target signal.

### 1.6 The target z must be defined for our RL model

LatentNet's `mse_z` computes `output_layer(x)` vs `z`. For the latentcircuit tutorial, `z` is a continuous supervised target (0.2 or 1.2 for left/right decision over time). Our model produces discrete action logits through `actor` head, not continuous targets.

Two options:
1. **Use actor logits as z**: After each timestep, record `actor_logits` (shape: 3) as z. This is a natural analog — LatentNet learns to predict what the actor head produces from latent states.
2. **Use action probabilities (softmax) as z**: Softer targets, may be easier to fit.

The output_layer of LatentNet must be sized to match z. If z = actor_logits, output_size=3. If z = softmax probs, output_size=3. Either works; actor logits are more direct.

**Key decision for implementation:** Define z = actor_logits per timestep. This means `output_size=3` for LatentNet.

### 1.7 LatentNet input_size constraint: diagonal connectivity mask

From `latent_net.py` `connectivity_masks()`:
```python
input_mask[:self.input_size, :self.input_size] = torch.eye(self.input_size)
output_mask[-self.output_size:, -self.output_size:] = torch.eye(self.output_size)
```

This enforces:
- `w_in` is diagonal in the top-left `(input_size x input_size)` block — requires `n >= input_size`
- `w_out` is diagonal in the bottom-right `(output_size x output_size)` block — requires `n >= output_size`

For our case:
- `input_size = 7` (ActorCritic input_dim: 5 obs + 1 context + 1 reward)
- `output_size = 3` (actor logits)
- `n` must be >= max(input_size, output_size) = 7

So **n >= 7** is required. Starting rank of n=8 satisfies this constraint.

### 1.8 NaN padding in hidden states must be stripped before fitting

`extract_behavior_with_hidden` pads shorter trials with NaN to create a uniform `(n_trials, max_T, 64)` array. LatentNet's `fit()` loops over time in its forward pass and does not tolerate NaN. All trials must be either:
- Truncated to the minimum observed trial length (lossy), or
- Processed trial-by-trial without padding (requires restructuring the DataLoader call), or
- Padded with zeros instead of NaN and a masking approach applied

The cleanest approach: re-collect hidden states with NaN-to-zero conversion, or — better — re-run extraction in fixed-length trial mode (neurogym trials of fixed length via timing override).

**This is a critical adapter issue.** LatentNet's `fit()` sends a full `(n_trials, T, n)` tensor through its forward pass — it cannot handle ragged sequences.

---

## 2. Data Format Specification

### 2.1 LatentNet expected inputs

| Argument | Type | Shape | Description |
|----------|------|-------|-------------|
| `u` | `torch.Tensor` (float) | `(n_trials, T, input_size)` | Input stimulus at each timestep |
| `z` | `torch.Tensor` (float) | `(n_trials, T, output_size)` | Target output at each timestep |
| `y` | `torch.Tensor` (float) | `(n_trials, T, N)` | RNN hidden states |

All three must be **the same (n_trials, T)** — uniform trial length, no NaN.

### 2.2 LatentNet constructor signature

```python
LatentNet(
    n=8,          # latent rank — must be >= max(input_size, output_size)
    N=64,         # full RNN hidden dim — must match y.shape[2]
    input_size=7, # must match u.shape[2]
    n_trials=..., # u.shape[0] (used only in __init__, not actually in forward)
    sigma_rec=0.15,  # noise level (same as in Tutorial)
    output_size=3,   # must match z.shape[2]
    device='cpu',
)
```

Note: `n_trials` parameter in `__init__` is stored as `self.n_trials` but is **not used** in `forward()` or `fit()`. It is vestigial. Pass `u.shape[0]` for correctness.

### 2.3 Our ActorCritic dimensions (ContextDecisionMaking-v0, dim_ring=2)

| Quantity | Value | Source |
|----------|-------|--------|
| obs_dim | 5 | fixation(1) + stim1_mod1,stim2_mod1(2) + stim1_mod2,stim2_mod2(2) |
| action_dim | 3 | fixation(0) + choice_1(1) + choice_2(2) |
| context_dim (single-task) | 1 | `env.set_num_tasks(1)` → context=[1.0] |
| reward_dim | 1 | appended to state |
| ActorCritic input_dim | 7 | 5 + 1 + 1 |
| ActorCritic hidden_dim | 64 | from Phase 2 training |

### 2.4 Trial length (ContextDecisionMaking-v0, dt=100ms)

| Period | Duration (ms) | Steps (dt=100) |
|--------|--------------|----------------|
| fixation | 300 | 3 |
| stimulus | 750 | 7-8 |
| delay | TruncExp(600, 300, 3000) | 3 to 30, mean ~6 |
| decision | 100 | 1 |
| **Total** | **variable** | **~14 to 41 steps** |

Phase 2 uses `max_steps=500` in training — actual trials are much shorter. Variable trial length means hidden states from Phase 2 have **variable actual lengths** within the NaN-padded array.

**Recommended solution for LatentNet:** Use fixed-length trials by setting `timing={"fixation": 300, "stimulus": 750, "delay": 0, "decision": 100}` during extraction, giving exactly ~12 steps per trial. This eliminates the padding problem.

Alternatively: truncate all trials to `min(trial_lengths)` before fitting.

### 2.5 Required adapter transformation

```
Phase 2 output:
  result['hidden']       → (n_trials, max_T, 64)  NaN-padded   → y
  [not captured]         → input per timestep                   → u
  [not captured]         → actor output per timestep            → z

Adapter must:
  1. Re-run the model (or re-instrument extraction) to also capture u and z
  2. Ensure uniform trial length (no NaN)
  3. Convert all to torch.Tensor float32
  4. Pass to LatentNet.fit(u, z, y, ...)
```

The Phase 2 script saves `hidden_context_dm.npy` but does NOT save `u` (inputs) or `z` (actor logits). The extraction must be re-run or extended.

---

## 3. Fitting Parameters (from paper/tutorial)

| Parameter | Tutorial value | Notes |
|-----------|---------------|-------|
| epochs | 500 | Per initialization |
| lr | 0.02 | Adam optimizer |
| l_y | 1 | Weight on hidden-state reconstruction term |
| weight_decay | 0.001 | Adam weight decay |
| batch_size | 128 | Hardcoded in `fit()` — cannot be changed without modifying LatentNet |
| n (latent rank) | 8 | For 50-unit RNN; start with 8 for our 64-unit case |
| sigma_rec | 0.15 | Noise during LatentNet forward pass |
| n_inits | 100 | Ensemble size (project decision) |

### 3.1 Selection criterion

Select the initialization with **lowest `nmse_y`** (normalized MSE between `x @ Q` and `y`) at the end of 500 epochs. This is what the roadmap calls "best solution by reconstruction loss."

`nmse_y` is computed per-epoch in the tutorial printout. After fitting, recompute:
```python
x = latent_net(u)
nmse_y_final = latent_net.nmse_y(y, x).item()
```

---

## 4. Adapter Requirements (ActorCritic → LatentNet)

### 4.1 What must be built: `collect_circuit_data()`

A new function (NOT `extract_behavior_with_hidden`) that runs the trained ActorCritic on ContextDecisionMaking-v0 and simultaneously collects:
- `u_list`: input tensor at each step — `torch.tensor(state)` before each `model()` call
- `y_list`: hidden state at each step — `h.squeeze()` after each `model()` call
- `z_list`: actor logits at each step — `actor_logits` after each `model()` call

Returns three uniform-length tensors ready for `LatentNet.fit()`.

**Key design decision:** Use fixed-length trial mode (no variable-length delay) or post-hoc truncation to the minimum trial length observed, to produce uniform `T`.

### 4.2 Weight extraction

After fitting, weights are accessed as:
```python
w_rec = latent_net.recurrent_layer.weight.data  # (n, n)
w_in  = latent_net.input_layer.weight.data       # (n, input_size)
w_out = latent_net.output_layer.weight.data      # (output_size, n)
Q     = latent_net.q                             # (n, N)
```

The RNN's full recurrent weight is:
```python
W_rec = model.rnn.weight_hh_l0  # (hidden_dim, hidden_dim) = (64, 64)
```

Note: `nn.RNN` with `batch_first=True` stores weights as `weight_hh_l0` (hidden-to-hidden) and `weight_ih_l0` (input-to-hidden). The convention in `nn.Linear` is `weight @ x` (rows = outputs, columns = inputs), but `nn.RNN` applies the weight as `h @ W_hh^T + x @ W_ih^T`. This means the effective recurrent matrix is `W_rec.T` in standard notation. **Must verify orientation before computing Q^T W_rec Q.**

### 4.3 Connectivity mask compatibility

LatentNet's `connectivity_masks()` enforces:
- `w_in` is identity-like in top-left block (only first `input_size` latent units receive input)
- `w_out` reads only from the last `output_size` latent units

With n=8, input_size=7, output_size=3:
- First 7 latent units receive inputs (nearly all of them)
- Last 3 latent units produce outputs

This reduces the interpretability somewhat but is structurally valid.

### 4.4 LatentNet's ReLU vs ActorCritic's tanh

LatentNet uses `torch.nn.ReLU()` internally. Our ActorCritic uses tanh. This mismatch is intentional and expected — LatentNet does not need to replicate the activation function of the source RNN. It learns to approximate the hidden state dynamics regardless of the source activation. The Tutorial notebook demonstrates this — their `Net` also uses ReLU, but the principle holds for any source RNN.

---

## 5. Pitfalls and Constraints

### 5.1 CRITICAL: u and z not saved by Phase 2

Phase 2's `extract_and_save()` saves `hidden_context_dm.npy` and `trial_lengths_context_dm.npy` but does NOT save the input sequence `u` or the actor output `z`. The Phase 3 pipeline cannot use Phase 2's saved outputs directly. Two options:

- **Option A (recommended):** Write a separate data collection function in `circuit_inference.py` that runs the model and captures u, y, z in one pass. This is self-contained and avoids modifying Phase 2 artifacts.
- **Option B:** Modify `train_context_dm.py`'s `extract_and_save()` to also save u and z. But this requires re-running Phase 2, which is already complete.

**Decision should be Option A** — new collection function in Phase 3 that reads the saved model checkpoint and runs a fresh collection pass.

### 5.2 CRITICAL: Variable trial length → NaN padding breaks LatentNet

LatentNet's `forward()` loops over `t = u.shape[1]` timesteps. If u contains NaN (from padding), the forward pass silently produces NaN in all subsequent states via the continuous-time update rule. The result will be NaN loss and broken gradients.

**Solution:** Collect data with a fixed-length trial configuration (set `timing={"delay": 0}` or similar) to get uniform ~12-step trials. Or collect only the minimum-length trials. Document the chosen T explicitly in the output metadata.

### 5.3 MODERATE: LatentNet batch_size=128 hardcoded

`fit()` creates a DataLoader with `batch_size=128`. If `n_trials < 128`, only one incomplete batch per epoch is processed. If `n_trials > 128`, multiple batches per epoch. For reliable fitting, collect at least 256-512 trials for each modality context.

### 5.4 MODERATE: 100 initializations × 500 epochs = 50,000 fitting epochs total

Each epoch with 500 trials and batch_size=128 processes ~4 batches. Total gradient steps ≈ 100 × 500 × 4 = 200,000. Each step involves forward + backward through the full trial sequence (12+ timesteps) for 128 trials at a time. On CPU with N=64, n=8, T=12: this should be manageable in under 2 hours, but needs profiling. The sequential constraint (16GB RAM) was already decided.

### 5.5 MODERATE: nn.RNN weight matrix orientation

`torch.nn.RNN` stores `weight_hh_l0` with shape `(hidden_dim, hidden_dim)`. The update rule is `h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1})`. So `W_hh` is applied as `weight_hh @ h` — the PyTorch convention for `nn.Linear` stores weights as `(out_features, in_features)`, so `weight_hh` has the same orientation. For computing `Q^T W_rec Q`, use `model.rnn.weight_hh_l0` directly (shape 64x64, row=output, col=input). This is consistent with how `latent_net.recurrent_layer.weight` is oriented (`nn.Linear` without bias).

### 5.6 MINOR: Q matrix is recomputed on every forward/backward call

In `fit()`:
```python
self.q = self.cayley_transform(self.a)
```
This is recomputed after every gradient update. `self.q` is not a `nn.Parameter` — it is recomputed from `self.a` (which IS a parameter). This means `Q` is always guaranteed orthonormal, but it means `self.q` should be re-accessed after fitting (do not cache the reference before `fit()` returns).

### 5.7 MINOR: LatentNet installed from local path dependency

The latentcircuit repo is at `C:\Users\aman0087\Documents\Github\latentcircuit`. To use it in our project:
```
pip install -e C:/Users/aman0087/Documents/Github/latentcircuit
```
or add to pyproject.toml dependencies. However, the latentcircuit repo does not have a `setup.py` or `pyproject.toml` — it is a flat Python module directory. The cleanest approach is to copy `latent_net.py` and `connectivity.py` into our package (e.g., `src/nn4psych/analysis/`) rather than trying to install it as a package. This avoids a fragile local path dependency.

**Recommended:** Copy `latent_net.py` → `src/nn4psych/analysis/latent_net.py` and import from there. No `connectivity.py` is needed since our code does not use `init_connectivity()` (Dale's law is for their `Net`, not `LatentNet`).

### 5.8 MINOR: Connectivity masks assume n >= input_size AND n >= output_size

From `connectivity_masks()`:
```python
input_mask[:self.input_size, :self.input_size] = torch.eye(self.input_size)
output_mask[-self.output_size:, -self.output_size:] = torch.eye(self.output_size)
```
If n=8, input_size=7, output_size=3:
- input_mask is (n, input_size) = (8, 7), and sets top 7x7 to identity → OK
- output_mask is (output_size, n) = (3, 8), and sets last 3x3 to identity → OK

But if n=7, input_size=7, output_size=3:
- output_mask sets rows [-3:, -3:] = [-3:, 4:7] → still OK since output_size=3 < n=7

And if n < input_size or n < output_size, the mask indexing will error. So **n must be >= max(input_size, output_size) = 7**. Starting with n=8 is safe.

---

## 6. Open Questions (Unresolved)

### 6.1 Exact paper rank for context-DM (LOW confidence, needs verification)

The tutorial uses n=8 for a 50-unit RNN. The paper (Langdon & Engel 2025) may specify a particular rank for the context-DM task. The decision context (03-CONTEXT.md) says "researcher looks up their choice for context-DM, likely 3-5 dims." This contradicts the tutorial's n=8 example. The paper should be consulted to find the exact rank used for the Mante/context-DM task. However, for planning purposes: **treat n=8 as the default starting point** and run the invariant subspace correlation check to validate. The 0.85 threshold will catch an insufficient rank.

### 6.2 Whether to use actor logits or action probabilities as z

Using raw actor logits (unbounded) as z means LatentNet's output layer learns to produce large values. Using softmax probabilities (bounded [0,1]) may give a better-conditioned optimization. This is a design decision with no clear answer from the tutorial. The tutorial uses a continuous supervised target (0.2 to 1.2) which is bounded. Using softmax probs as z may be more consistent with that approach.

### 6.3 Whether both modality contexts (0 and 1) should be used jointly or separately

03-CONTEXT.md says "both modality contexts needed." This means the model must be trained on both `modality_context=0` and `modality_context=1`, and hidden states collected for both. But LatentNet.fit() takes a single batch of (u, z, y). Two design choices:
- **Option A (joint):** Concatenate trials from both modality contexts into one batch for fitting. LatentNet learns a circuit that captures both.
- **Option B (separate):** Fit separate LatentNet instances for each context. Compare the two resulting circuits.

The paper's context-DM analysis uses context-dependent routing — the context-dependent latent space is only revealed when both contexts are present in the data. **Option A (joint fitting) is the correct approach** consistent with the paper's framing. The context signal should be included in `u` (which it is — our ActorCritic input includes context=[1.0] for single-task; for dual-context extraction, context would be a 2-dim one-hot or a single bit).

This requires re-training the ActorCritic on both contexts (as noted in 03-CONTEXT.md decision: "retrain model on both modality_context=0 and modality_context=1"). The current Phase 2 script trains on a single modality_context.

### 6.4 Perturbation methodology details

The roadmap requires: "perturbing w_rec in latent space and mapping back to RNN weights." The exact mapping is:
```
W_rec_perturbed = W_rec + Q^T @ delta_w_rec @ Q
```
where `delta_w_rec` is a small perturbation in the latent space. This projects the latent perturbation back into the full 64x64 weight space. The Tutorial.ipynb does NOT demonstrate perturbation analysis — it only shows fit quality checks. The perturbation methodology must be derived from the paper text directly.

**Risk:** The paper may use a different perturbation convention (e.g., rank-1 perturbations, specific connectivity targets). The implementation in 03-03 will need to read the paper carefully.

---

## Appendix: Tutorial Workflow (verbatim from Tutorial.ipynb)

```python
# 1. Generate task data
u, z, mask, conditions = generate_trials(n_trials=25, alpha=0.2, sigma_in=0.01, baseline=0.2, n_coh=6)
# u: (n_trials, 75, 6), z: (n_trials, 75, 2), mask: (n_trials, 75, 2)

# 2. Train their RNN (we skip this — we use our ActorCritic from Phase 2)
net = Net(n=50, input_size=6, dale=True, sigma_rec=0.15)
net.fit(u, z, mask, lr=.01, epochs=150, verbose=True, weight_decay=0.001)

# 3. Simulate RNN to get hidden states
y = net(u)           # (n_trials, 75, 50)
z = net.output_layer(y)  # (n_trials, 75, 2) — re-derive z from trained net

# 4. Initialize and fit LatentNet
latent_net = LatentNet(n=8, N=net.n, input_size=6, n_trials=u.shape[0], sigma_rec=0.15)
loss_history = latent_net.fit(u.detach(), z.detach(), y.detach(),
                              epochs=500, lr=.02, l_y=1, weight_decay=0.001)

# 5. Validate: activity level
qx = latent_net(u) @ latent_net.q   # (n_trials, 75, 50) projected latent → full space
# scatter qx vs y → should lie on diagonal

# 6. Validate: connectivity level (invariant subspace)
w_rec = latent_net.recurrent_layer.weight.data.detach()   # (8, 8)
q = latent_net.q.detach()                                  # (8, 50)
W_rec = net.recurrent_layer.weight.data.detach()           # (50, 50)
# q @ W_rec @ q.T should visually match w_rec
# Compute correlation: corr(w_rec.flatten(), (q @ W_rec @ q.T).flatten())
```

Our adapter must produce equivalent `u`, `z`, `y` from the ActorCritic / ContextDecisionMaking pipeline.
